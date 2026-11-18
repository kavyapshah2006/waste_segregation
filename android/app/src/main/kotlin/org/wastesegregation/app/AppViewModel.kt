package org.wastesegregation.app

import android.app.Application
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.Settings
import android.speech.tts.TextToSpeech
import androidx.camera.core.ImageProxy
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import wastesegregation.core.Assistant
import wastesegregation.core.AttributeChecker
import wastesegregation.core.Correction
import wastesegregation.core.CorrectionLog
import wastesegregation.core.Detected
import wastesegregation.core.Education
import wastesegregation.core.FrameVoter
import wastesegregation.core.Impact
import wastesegregation.core.Json
import wastesegregation.core.JsonNumber
import wastesegregation.core.KnowledgeBase
import wastesegregation.core.LatencyLog

import wastesegregation.core.Scan
import wastesegregation.core.Stages
import wastesegregation.core.Suggestion
import wastesegregation.core.Verdict
import wastesegregation.core.VerifiedOutput
import wastesegregation.core.Verifier
import wastesegregation.core.Voice
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

data class UiState(
    val status: String = "Loading…",
    val models: Map<String, String> = emptyMap(),
    val language: String = "en",
    val scanState: String = "empty",
    val liveItem: String? = null,
    val verdict: Verdict? = null,
    val confidence: Double? = null,
    val attributes: Map<String, Any?> = emptyMap(),
    val askAttributes: List<String> = emptyList(),
    val explaining: Boolean = false,
    val whyStream: String = "",
    val verified: VerifiedOutput? = null,
    val question: String? = null,
    val suggestions: List<Suggestion> = emptyList(),
    val unsureFrames: Int = 0,
    val detectMs: Double? = null,
    val cardMs: Double? = null,
    val firstTokenS: Double? = null,
    val decodeTps: Double? = null,
    val showOverlay: Boolean = true,
    val impact: Map<String, Any?>? = null,
    val didYouKnow: Map<String, Any?>? = null,
    val quiz: Map<String, Any?>? = null,
    val message: String? = null,
    val listening: Boolean = false,
    val transcribing: Boolean = false,
    val transcript: String? = null,
    val voiceChoices: List<String> = emptyList(),
)

/** Everything the screens need: models, the scan state machine, Llama, stores. */
@Suppress("UNCHECKED_CAST")
class AppViewModel(app: Application) : AndroidViewModel(app) {
    var ui by mutableStateOf(UiState())
        private set

    private val main = Handler(Looper.getMainLooper())
    val analysisExecutor = Executors.newSingleThreadExecutor()
    private val work = Executors.newSingleThreadExecutor()
    private val voiceThread = Executors.newSingleThreadExecutor()

    private val files = ModelFiles(app)
    private var kb: KnowledgeBase? = null
    private var detector: Detector? = null
    private var siglip: Siglip? = null
    private var llama: Llama? = null
    private var checker: AttributeChecker? = null
    private var speech: SpeechInput? = null
    private val voter = FrameVoter()
    val latency = LatencyLog()
    private val llmStats = mutableListOf<String>()
    private val scans = ScanStore(File(app.filesDir, "scans.jsonl"))
    private val corrections = CorrectionLog(File(app.filesDir, "corrections.jsonl"))
    private var tts: TextToSpeech? = null

    @Volatile private var locked = false
    @Volatile private var llmBusy = false
    @Volatile private var lastFrame: Bitmap? = null

    /** State changes always land on the main thread; callers may be on any thread. */
    private fun update(f: (UiState) -> UiState) {
        main.post { ui = f(ui) }
    }

    init {
        tts = TextToSpeech(app) { }
        work.execute { loadModels() }
    }

    private fun loadModels() {
        val status = linkedMapOf<String, String>()
        fun report() = update { it.copy(models = LinkedHashMap(status)) }
        try {
            kb = files.kb()?.let { KbLoader.load(it) }
            status["Knowledge base"] = kb?.let { "ready (${it.itemIds.size} items)" } ?: "missing"
        } catch (e: Exception) {
            status["Knowledge base"] = "error: ${e.message}"
        }
        report()
        val kb = kb ?: return update { it.copy(status = "Knowledge base missing") }
        status["Detector"] = try {
            val pte = files.detector()
            val names = files.detectorNames()
            if (pte != null && names != null) {
                detector = Detector(pte, ModelFiles.classNames(names), kb)
                "ready"
            } else "missing (train and export the detector)"
        } catch (e: Exception) {
            "error: ${e.message}"
        }
        report()
        status["Dirty/broken check"] = try {
            files.attributePrompts()?.let { checker = AttributeChecker(Json.parse(it.readText()) as Map<String, Any?>); "ready" }
                ?: "missing"
        } catch (e: Exception) {
            "error: ${e.message}"
        }
        status["Unknown-item fallback"] = try {
            val pte = files.siglip()
            val text = files.textEmbeddings()
            if (pte != null && text != null) {
                siglip = Siglip(pte, text)
                "ready"
            } else "missing (push siglip2_image_int8.pte)"
        } catch (e: Exception) {
            "error: ${e.message}"
        }
        status["Voice input"] = run {
            val model = files.asrModel()
            val tokens = files.asrTokens()
            if (model != null && tokens != null) {
                speech = SpeechInput(model, tokens)
                "ready (loads per question)"
            } else "missing (push model.int8.onnx and tokens.txt)"
        }
        report()
        status["Llama"] = try {
            val model = files.llama()
            val tok = files.tokenizer()
            if (model != null && tok != null) {
                status["Llama"] = "loading ${model.name}…"
                report()
                val t0 = SystemClock.elapsedRealtime()
                llama = Llama(model, tok).also { it.load() }
                "ready: ${model.name} (load ${(SystemClock.elapsedRealtime() - t0) / 1000.0} s)"
            } else "missing (push the Llama .pte and tokenizer.model)"
        } catch (e: Throwable) {
            "error: ${e.message}"
        }
        report()
        update { it.copy(status = "Point the camera at one item") }
        refreshImpact()
    }

    // ------------------------------------------------------------------ camera

    fun onFrame(proxy: ImageProxy) {
        val det = detector
        if (det == null || locked || llmBusy) {
            proxy.close()
            return
        }
        val t0 = SystemClock.elapsedRealtimeNanos()
        val bitmap = try {
            latency.time(Stages.CAPTURE) { proxy.toBitmap().rotate(proxy.imageInfo.rotationDegrees) }
        } finally {
            proxy.close()
        }
        lastFrame = bitmap
        val detections = det.detect(bitmap, latency)
        val state = latency.time(Stages.VOTE) {
            voter.push(detections.map { Detected(it.itemId, it.confidence.toDouble()) })
        }
        val detectMs = latency.samples(Stages.DETECT).lastOrNull()
        val name = state.itemId?.let { kb?.resolve(it, emptyMap(), ui.language)?.name }
        update {
            it.copy(
                scanState = state.state,
                liveItem = name,
                detectMs = detectMs,
                unsureFrames = if (state.state == "unsure") it.unsureFrames + 1 else 0,
            )
        }
        if (state.state == "stable" && state.itemId != null) {
            val best = detections.filter { it.itemId == state.itemId }.maxByOrNull { it.confidence }
            val crop = best?.let { bitmap.crop(it.bbox) } ?: bitmap
            lock(state.itemId!!, crop, state.confidence, t0)
        }
    }

    /**
     * Show the verdict for [itemId] at once (KB only), then ask Llama to explain. Runs on a
     * background thread (analysis or work): the attribute check runs the SigLIP encoder.
     */
    private fun lock(
        itemId: String,
        crop: Bitmap?,
        confidence: Double?,
        t0Nanos: Long,
        question: String? = null,
        language: String = ui.language,
    ) {
        val kb = kb ?: return
        locked = true
        val attributes = linkedMapOf<String, Any?>()
        val ask = mutableListOf<String>()
        val siglip = siglip
        val checker = checker
        if (crop != null && siglip != null && checker != null && checker.attributesFor(itemId).isNotEmpty()) {
            val emb = latency.time(Stages.ATTRIBUTES) { siglip.embed(crop) }
            for ((attr, result) in checker.check(itemId, emb)) {
                if (result.value == null) ask.add(attr) else attributes[attr] = result.value
            }
        }
        val verdict = latency.time(Stages.KB) { kb.resolve(itemId, attributes, language) }
        val cardMs = (SystemClock.elapsedRealtimeNanos() - t0Nanos) / 1e6
        latency.record(Stages.FRAME_TO_CARD, cardMs)
        scans.add(itemId, attributes)
        update {
            it.copy(
                verdict = verdict, confidence = confidence, attributes = attributes, askAttributes = ask,
                verified = null, whyStream = "", question = question, suggestions = emptyList(),
                cardMs = cardMs, scanState = "locked",
            )
        }
        speak(verdict)
        explain(question, verdict, attributes)
        refreshImpact()
    }

    fun scanAgain() {
        voter.reset()
        locked = false
        update {
            it.copy(verdict = null, verified = null, whyStream = "", suggestions = emptyList(), scanState = "empty",
                askAttributes = emptyList(), question = null)
        }
    }

    // ------------------------------------------------------------------ Llama

    fun explain(
        question: String?,
        verdictArg: Verdict? = ui.verdict,
        attributes: Map<String, Any?> = ui.attributes,
    ) {
        val kb = kb ?: return
        val verdict = verdictArg ?: return
        val llama = llama
        if (llama == null) {
            val verified = if (Assistant.isOnTopic(question, kb.namesOf(verdict.itemId))) {
                VerifiedOutput("fallback", Verifier.kbOutput(verdict), listOf("no_model"))
            } else Assistant.declined(kb, verdict)
            update { it.copy(verified = verified, question = question) }
            return
        }
        update { it.copy(explaining = true, whyStream = "", verified = null, question = question) }
        work.execute {
            llmBusy = true
            try {
                fun run(prompt: String, temperature: Float): String {
                    val start = SystemClock.elapsedRealtimeNanos()
                    var first = true
                    val raw = StringBuilder()
                    val text = llama.generate(prompt, temperature = temperature) { piece ->
                        if (first) {
                            first = false
                            latency.record(Stages.LLM_FIRST_TOKEN, (SystemClock.elapsedRealtimeNanos() - start) / 1e6)
                        }
                        raw.append(piece)
                        val partial = partialWhy(raw.toString())
                        update { it.copy(whyStream = partial) }
                    }
                    llama.lastStats?.let { s -> synchronized(llmStats) { llmStats.add(s) } }
                    return text
                }
                // Greedy first; sampled retries (temperature 0.7) only if the checks fail.
                val explanation = Assistant.explain(
                    kb, verdict, attributes, { run(it, 0f) }, question, sample = { run(it, 0.7f) },
                )
                val tps = llama.lastStats?.let { decodeTokensPerSecond(it) }
                update {
                    it.copy(verified = explanation.verified, explaining = false, decodeTps = tps,
                        firstTokenS = latency.samples(Stages.LLM_FIRST_TOKEN).lastOrNull()?.div(1000))
                }
            } catch (e: Throwable) {
                update { it.copy(explaining = false, message = "Llama failed: ${e.message}") }
            } finally {
                llmBusy = false
            }
        }
    }

    fun ask(question: String) {
        val kb = kb ?: return
        if (ui.verdict == null) {
            val match = kb.search(question, ui.language).firstOrNull() ?: kb.search(question).firstOrNull()
            if (match == null) {
                update { it.copy(message = "Scan an item first, or name it in your question.") }
                return
            }
            locked = true
            work.execute { lock(match, null, null, SystemClock.elapsedRealtimeNanos(), question) }
            return
        }
        explain(question)
    }

    // ------------------------------------------------------------------ user input

    fun setAttribute(attribute: String, value: Boolean) {
        val kb = kb ?: return
        val itemId = ui.verdict?.itemId ?: return
        val attributes = LinkedHashMap(ui.attributes).apply { put(attribute, value) }
        val verdict = kb.resolve(itemId, attributes, ui.language)
        update { it.copy(attributes = attributes, verdict = verdict, askAttributes = it.askAttributes - attribute) }
        explain(ui.question, verdict, attributes)
    }

    fun setLanguage(language: String) {
        val kb = kb ?: return
        update { it.copy(language = language) }
        val itemId = ui.verdict?.itemId ?: return
        val verdict = kb.resolve(itemId, ui.attributes, language)
        update { it.copy(verdict = verdict) }
        explain(ui.question, verdict, ui.attributes)
        refreshImpact()
    }

    /** Suggestions from the SigLIP fallback for the last frame (or a picked photo). */
    fun suggest(bitmap: Bitmap? = lastFrame) {
        val siglip = siglip
        if (siglip == null || bitmap == null) {
            update { it.copy(message = "Fallback model not loaded") }
            return
        }
        work.execute {
            val emb = latency.time(Stages.FALLBACK) { siglip.embed(bitmap) }
            val suggestions = siglip.suggest(emb)
            update { it.copy(suggestions = suggestions) }
        }
    }

    fun choose(itemId: String) {
        locked = true
        work.execute { lock(itemId, lastFrame, null, SystemClock.elapsedRealtimeNanos()) }
    }

    /** "This is wrong": log the correction (with the photo), then show the right item. */
    fun correct(itemId: String) {
        val wrong = ui.verdict
        val frame = lastFrame
        val now = System.currentTimeMillis()
        val image = frame?.let { saveJpeg(it, "corrections/$now.jpg") }
        corrections.append(
            Correction(
                timeMs = now,
                source = if (ui.confidence == null) "fallback" else "detector",
                predicted = wrong?.itemId,
                corrected = itemId,
                confidence = ui.confidence?.coerceIn(0.0, 1.0),
                attributes = ui.attributes.filterValues { it is Boolean }.mapValues { it.value as Boolean },
                image = image,
            ),
        )
        locked = true
        work.execute { lock(itemId, frame, null, SystemClock.elapsedRealtimeNanos()) }
    }

    fun photo(bitmap: Bitmap) {
        val det = detector
        lastFrame = bitmap
        work.execute {
            val t0 = SystemClock.elapsedRealtimeNanos()
            val best = det?.detect(bitmap, latency)?.filter { it.itemId != null }?.maxByOrNull { it.confidence }
            if (best != null && best.confidence >= GATE) {
                lock(best.itemId!!, bitmap.crop(best.bbox), best.confidence.toDouble(), t0)
            } else {
                locked = true
                update { it.copy(scanState = "unsure", verdict = null, message = "Not sure what this is: pick one") }
                suggest(bitmap)
            }
        }
    }

    // ------------------------------------------------------------------ voice

    /** Push-to-talk: record until [stopVoice] (or 8 s), transcribe on the phone, then act. */
    fun startVoice() {
        val speech = speech
        if (speech == null) {
            update { it.copy(message = "Voice model not on the phone (see docs/PHONE_SESSION.md)") }
            return
        }
        if (ui.listening || ui.transcribing) return
        update { it.copy(listening = true, transcript = null, voiceChoices = emptyList()) }
        voiceThread.execute {
            try {
                val audio = speech.record()
                update { it.copy(listening = false, transcribing = true) }
                val text = latency.time(Stages.ASR) { speech.transcribe(audio) }
                handleTranscript(text)
            } catch (e: Throwable) {
                update { it.copy(listening = false, transcribing = false, message = "Voice failed: ${e.message}") }
            }
        }
    }

    fun stopVoice() {
        speech?.stop()
    }

    private fun handleTranscript(text: String) {
        val kb = kb ?: return
        // Devanagari in the transcript: answer in Hindi.
        val language = if (text.any { it in '\u0900'..'\u097F' }) "hi" else ui.language
        update { it.copy(transcribing = false, transcript = text, language = language) }
        if (text.isBlank()) {
            update { it.copy(message = if (language == "hi") "कुछ सुनाई नहीं दिया, फिर से बोलें" else "Didn't catch that, try again") }
            return
        }
        val matches = Voice.match(kb, text)
        val top = matches.firstOrNull()
        when {
            top != null && !top.confirm -> {
                locked = true
                work.execute { lock(top.itemId, null, null, SystemClock.elapsedRealtimeNanos(), text, language) }
            }
            matches.isNotEmpty() -> update { it.copy(voiceChoices = matches.map { m -> m.itemId }) }
            ui.verdict != null -> {
                val verdict = kb.resolve(ui.verdict!!.itemId, ui.attributes, language)
                update { it.copy(verdict = verdict) }
                explain(text, verdict, ui.attributes)
            }
            else -> update { it.copy(message = "Say the item's name, e.g. \"old battery\"") }
        }
    }

    /** The user picked one of the voice matches: show it and answer the spoken question. */
    fun chooseVoice(itemId: String) {
        val text = ui.transcript
        val language = ui.language
        update { it.copy(voiceChoices = emptyList()) }
        locked = true
        work.execute { lock(itemId, null, null, SystemClock.elapsedRealtimeNanos(), text, language) }
    }

    fun itemChoices(): List<Pair<String, String>> {
        val kb = kb ?: return emptyList()
        return kb.itemIds.map { it to kb.resolve(it, emptyMap(), ui.language).name }.sortedBy { it.second }
    }

    fun nameOf(itemId: String): String = kb?.resolve(itemId, emptyMap(), ui.language)?.name ?: itemId

    fun toggleOverlay() = update { it.copy(showOverlay = !it.showOverlay) }

    fun clearMessage() = update { it.copy(message = null) }

    // ------------------------------------------------------------------ impact, speech, export

    fun refreshImpact() {
        val kb = kb ?: return
        work.execute {
            val now = System.currentTimeMillis() / 1000.0
            val tz = java.util.TimeZone.getDefault().getOffset(System.currentTimeMillis()) / 60000
            val day = Impact.localDay(now, tz)
            val summary = Impact.summary(kb, scans.all(), now, tz)
            update {
                it.copy(impact = summary, didYouKnow = Education.didYouKnow(kb, day, it.language),
                    quiz = Education.quiz(kb, day, it.language))
            }
        }
    }

    private fun speak(verdict: Verdict) {
        val tts = tts ?: return
        tts.language = if (verdict.language == "hi") Locale.forLanguageTag("hi-IN") else Locale.ENGLISH
        val line = if (verdict.language == "hi") {
            "${verdict.name}: ${verdict.binColourName} डिब्बा, ${verdict.streamName}"
        } else "${verdict.name}: ${verdict.binColourName} bin, ${verdict.streamName}"
        tts.speak(line, TextToSpeech.QUEUE_FLUSH, null, "verdict")
    }

    /** Writes the stage timings and Llama stats for eval/results (pull with adb). */
    fun exportTimings(): String {
        val app = getApplication<Application>()
        val dir = (app.getExternalFilesDir("results") ?: File(app.filesDir, "results")).apply { mkdirs() }
        val stamp = SimpleDateFormat("yyyy-MM-dd_HHmm", Locale.US).format(Date())
        val airplane = Settings.Global.getInt(app.contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) == 1
        val device = linkedMapOf<String, Any?>(
            "model" to "${Build.MANUFACTURER} ${Build.MODEL}",
            "soc" to if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE,
            "android" to Build.VERSION.RELEASE,
        )
        val meta = linkedMapOf<String, Any?>(
            "date" to stamp.substring(0, 10),
            "app_version" to app.packageManager.getPackageInfo(app.packageName, 0).versionName,
            "device" to device,
            "models" to linkedMapOf("detector" to files.detector()?.name, "llm" to files.llama()?.name,
                "fallback" to files.siglip()?.name),
            "conditions" to linkedMapOf("airplane_mode" to airplane),
        )
        val out = File(dir, "latency_$stamp.json")
        out.writeText(latency.toJson(meta) + "\n")
        val stats = synchronized(llmStats) { llmStats.toList() }
        File(dir, "llm_stats_$stamp.jsonl").writeText(stats.joinToString("\n") + "\n")
        return out.path
    }

    private fun saveJpeg(bitmap: Bitmap, relative: String): String {
        val app = getApplication<Application>()
        val file = File(app.filesDir, relative).apply { parentFile?.mkdirs() }
        file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG, 90, it) }
        return relative
    }

    override fun onCleared() {
        tts?.shutdown()
        llama?.close()
        detector?.close()
        siglip?.close()
        analysisExecutor.shutdown()
        work.shutdown()
        voiceThread.shutdown()
    }

    companion object {
        const val GATE = 0.5f

        /** The text of "why" so far, from the raw output after the answer prefix. */
        fun partialWhy(raw: String): String {
            val sb = StringBuilder()
            var i = 0
            while (i < raw.length) {
                val c = raw[i]
                if (c == '\\' && i + 1 < raw.length) {
                    sb.append(if (raw[i + 1] == 'n') '\n' else raw[i + 1])
                    i += 2
                    continue
                }
                if (c == '"') break
                sb.append(c)
                i++
            }
            return sb.toString()
        }

        fun decodeTokensPerSecond(stats: String): Double? = try {
            val m = Json.parse(stats) as Map<String, Any?>
            val generated = (m["generated_tokens"] as JsonNumber).text.toDouble()
            val firstToken = (m["first_token_ms"] as JsonNumber).text.toDouble()
            val end = (m["inference_end_ms"] as JsonNumber).text.toDouble()
            if (end > firstToken) (generated - 1) / ((end - firstToken) / 1000) else null
        } catch (e: Exception) {
            null
        }
    }
}

private fun Bitmap.rotate(degrees: Int): Bitmap {
    if (degrees == 0) return this
    val m = Matrix().apply { postRotate(degrees.toFloat()) }
    return Bitmap.createBitmap(this, 0, 0, width, height, m, true)
}

/** Crop a normalised [x1, y1, x2, y2] box with a 10% margin (as models/fallback/check_crops.py). */
private fun Bitmap.crop(box: List<Float>): Bitmap {
    val mx = 0.1f * (box[2] - box[0])
    val my = 0.1f * (box[3] - box[1])
    val x1 = ((box[0] - mx) * width).toInt().coerceIn(0, width - 1)
    val y1 = ((box[1] - my) * height).toInt().coerceIn(0, height - 1)
    val x2 = ((box[2] + mx) * width).toInt().coerceIn(x1 + 1, width)
    val y2 = ((box[3] + my) * height).toInt().coerceIn(y1 + 1, height)
    return Bitmap.createBitmap(this, x1, y1, x2 - x1, y2 - y1)
}

/** Scan history for the impact screen (JSON lines in internal storage). */
@Suppress("UNCHECKED_CAST")
class ScanStore(private val file: File) {
    @Synchronized
    fun add(itemId: String, attributes: Map<String, Any?>) {
        val row = linkedMapOf("time_ms" to System.currentTimeMillis(), "item_id" to itemId,
            "attributes" to attributes.filterValues { it is Boolean })
        file.appendText(Json.dumps(row) + "\n")
    }

    @Synchronized
    fun all(): List<Scan> {
        if (!file.exists()) return emptyList()
        return file.readLines().filter { it.isNotBlank() }.mapNotNull { line ->
            try {
                val m = Json.parse(line) as Map<String, Any?>
                Scan((m["time_ms"] as JsonNumber).text.toLong() / 1000.0, m["item_id"] as String,
                    m["attributes"] as Map<String, Any?>)
            } catch (e: Exception) {
                null
            }
        }
    }
}
