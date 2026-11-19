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
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
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
import wastesegregation.core.FallbackRank
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

/** One past scan, ready to show in History. */
data class HistoryRow(val timeMs: Long, val itemId: String, val name: String, val binColour: String,
    val binColourName: String, val streamName: String)

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
    /** Nothing stable for a second: offer the "Suggest" button. */
    val canSuggest: Boolean = false,
    /** SigLIP is ranking KB items for the current frame. */
    val lookingCloser: Boolean = false,
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
    /** The photo (detector crop or chosen frame) shown on the result card. */
    val verdictImage: Bitmap? = null,
    val history: List<HistoryRow> = emptyList(),
    val torchOn: Boolean = false,
    /** Llama is loading for its first explanation (~30 s on a budget phone). */
    val llamaLoading: Boolean = false,
    /** "12 / 70" while Settings > Prepare explanations runs. */
    val preparing: String? = null,
    /** Explanations saved on this phone. */
    val cachedAnswers: Int = 0,
)

/** Everything the screens need: models, the scan state machine, Llama, stores. */
@Suppress("UNCHECKED_CAST")
class AppViewModel(app: Application) : AndroidViewModel(app) {
    var ui by mutableStateOf(UiState())
        private set

    private val main = Handler(Looper.getMainLooper())
    val analysisExecutor = Executors.newSingleThreadExecutor()
    private val work = Executors.newSingleThreadExecutor()
    /** SigLIP (suggestions, picks) gets its own thread so it never waits behind Llama. */
    private val vision = Executors.newSingleThreadExecutor()
    /** Shutter photos are copied here the moment they arrive, so the camera buffer is never
     *  held while the vision thread waits for another model (it was, and got closed under us). */
    private val captureThread = Executors.newSingleThreadExecutor()
    /** A shutter photo is being taken or analysed: further taps are ignored until it is done. */
    @Volatile private var capturing = false
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
    private val explanations = ExplanationCache(File(app.filesDir, "explanations.jsonl"))
    /** Llama files on the phone; the model itself loads on the first explanation. */
    @Volatile private var llamaFiles: Pair<File, File>? = null
    @Volatile private var preparing = false
    private var tts: TextToSpeech? = null

    @Volatile private var locked = false
    /** All models loaded (or given up on). Scanning waits for it: models run one at a time, so
     *  a scan during Llama's ~30 s load would only sit waiting behind it. */
    @Volatile private var ready = false
    @Volatile private var llmBusy = false
    @Volatile private var lastFrame: Bitmap? = null
    /** When the current search started (first frame without a stable item), for auto-suggest. */
    @Volatile private var searchStartMs: Long? = null
    /** Bumped to cancel a running explanation (scan again, new item). */
    @Volatile private var explainGeneration = 0

    private class Cancelled : RuntimeException()

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
                siglip = Siglip(pte, text).also { model ->
                    // First inference allocates and plans; do it now, not when the user waits.
                    runCatching { model.embed(Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)) }
                }
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
        // Llama loads on the first explanation, not here: scanning is ready in seconds instead of
        // waiting ~30 s, and cached explanations never need Llama at all.
        status["Llama"] = run {
            val model = files.llama()
            val tok = files.tokenizer()
            if (model != null && tok != null) {
                llamaFiles = model to tok
                "on the phone; loads on the first explanation (~30 s)"
            } else "missing (push the Llama .pte and tokenizer.model)"
        }
        report()
        ready = true
        update { it.copy(status = "Point the camera at one item", cachedAnswers = explanations.size()) }
        refreshImpact()
    }

    // ------------------------------------------------------------------ camera

    fun onFrame(proxy: ImageProxy) {
        val det = detector
        if (det == null || !ready || locked || llmBusy) {
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
        val now = SystemClock.elapsedRealtime()
        val searching = state.state != "stable"
        if (!searching) searchStartMs = null else if (searchStartMs == null) searchStartMs = now
        val waited = searchStartMs?.let { now - it } ?: 0L
        update {
            it.copy(
                scanState = state.state,
                liveItem = name,
                detectMs = detectMs,
                canSuggest = searching && waited >= SUGGEST_BUTTON_MS,
            )
        }
        // The detector sees something but cannot decide: ask SigLIP straight away rather than
        // making the user wait and tap. (An empty view only gets the button: nothing to rank.)
        if (state.state == "unsure" && waited >= AUTO_SUGGEST_MS) {
            suggest(bitmap)
            return
        }
        if (state.state == "stable" && state.itemId != null) {
            if (!agrees(state.itemId!!, bitmap)) return
            val best = detections.filter { it.itemId == state.itemId }.maxByOrNull { it.confidence }
            val crop = best?.let { bitmap.crop(it.bbox) } ?: bitmap
            lock(state.itemId!!, crop, state.confidence, t0)
        }
    }

    /**
     * Agreement rule: the detector's item becomes a verdict only if SigLIP also ranks it among
     * its top 3 for the same frame. Otherwise show those three as suggestions (no verdict) and
     * return false. The detector alone calls fruit peels and fish waste "carry bag"; on 2,371
     * BDWaste photos this rule lifts the right bin from 75% to 92% (eval/bdwaste/evaluate.py).
     * Without the SigLIP model the detector decides alone, as before.
     */
    private fun agrees(itemId: String, frame: Bitmap): Boolean {
        val siglip = siglip ?: return true
        update { it.copy(lookingCloser = true) }
        val top3 = siglip.suggest(siglip.embed(frame))
        latency.record(Stages.AGREEMENT, siglip.lastMs)
        update { it.copy(lookingCloser = false) }
        if (FallbackRank.confirms(itemId, top3)) return true
        locked = true
        lastFrame = frame
        update { it.copy(scanState = "unsure", verdict = null, suggestions = top3, lookingCloser = false) }
        return false
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
            val emb = siglip.embed(crop).also { latency.record(Stages.ATTRIBUTES, siglip.lastMs) }
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
                cardMs = cardMs, scanState = "locked", verdictImage = crop, lookingCloser = false,
            )
        }
        speak(verdict)
        explain(question, verdict, attributes)
        refreshImpact()
    }

    fun scanAgain() {
        cancelExplanation()
        capturing = false
        voter.reset()
        searchStartMs = null
        locked = false
        update {
            it.copy(verdict = null, verified = null, whyStream = "", suggestions = emptyList(), scanState = "empty",
                askAttributes = emptyList(), question = null, explaining = false, lookingCloser = false,
                canSuggest = false, verdictImage = null)
        }
    }

    /** Stop a running explanation: the camera waits for Llama (memory), so never keep it waiting. */
    private fun cancelExplanation() {
        explainGeneration++
        if (llmBusy) runCatching { llama?.stop() }
    }

    // ------------------------------------------------------------------ Llama

    fun explain(
        question: String?,
        verdictArg: Verdict? = ui.verdict,
        attributes: Map<String, Any?> = ui.attributes,
    ) {
        val kb = kb ?: return
        val verdict = verdictArg ?: return
        // Plain scans reuse what Llama already wrote on this phone for the same item and condition.
        val key = if (question == null) ExplanationCache.key(kb.version, verdict.itemId, attributes, verdict.language) else null
        val cached = key?.let { explanations.get(it) }
        if (cached != null) {
            explainGeneration++
            update { it.copy(verified = cached, explaining = false, whyStream = "", question = null) }
            return
        }
        if (llama == null && llamaFiles == null) {
            val verified = if (Assistant.isOnTopic(question, kb.namesOf(verdict.itemId))) {
                VerifiedOutput("fallback", Verifier.kbOutput(verdict), listOf("no_model"))
            } else Assistant.declined(kb, verdict)
            update { it.copy(verified = verified, question = question) }
            return
        }
        val gen = ++explainGeneration
        update { it.copy(explaining = true, whyStream = "", verified = null, question = question) }
        val start = Runnable {
            if (gen == explainGeneration) work.execute { runExplanation(gen, kb, verdict, attributes, question, key) }
        }
        // A scan waits a moment before Llama starts: someone who glances at the bin and scans the
        // next item never waits behind Llama reading its prompt (which cannot be interrupted).
        if (question == null) main.postDelayed(start, EXPLAIN_DELAY_MS) else start.run()
    }

    private fun runExplanation(
        gen: Int,
        kb: KnowledgeBase,
        verdict: Verdict,
        attributes: Map<String, Any?>,
        question: String?,
        key: String?,
    ) {
        if (gen != explainGeneration) return
        llmBusy = true
        try {
            val llama = ensureLlama()
            if (llama == null) {
                update { it.copy(explaining = false, verified = VerifiedOutput("fallback", Verifier.kbOutput(verdict), listOf("no_model"))) }
                return
            }
            fun run(prompt: String, temperature: Float): String {
                if (gen != explainGeneration) throw Cancelled()
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
                    if (gen == explainGeneration) update { it.copy(whyStream = partial) }
                }
                llama.lastStats?.let { s -> synchronized(llmStats) { llmStats.add(s) } }
                if (gen != explainGeneration) throw Cancelled()
                return text
            }
            // Greedy first; sampled retries (temperature 0.7) only if the checks fail.
            val explanation = Assistant.explain(
                kb, verdict, attributes, { run(it, 0f) }, question, sample = { run(it, 0.7f) },
            )
            key?.let { explanations.put(it, explanation.verified) }
            val tps = llama.lastStats?.let { decodeTokensPerSecond(it) }
            update {
                it.copy(verified = explanation.verified, explaining = false, decodeTps = tps,
                    firstTokenS = latency.samples(Stages.LLM_FIRST_TOKEN).lastOrNull()?.div(1000),
                    cachedAnswers = explanations.size())
            }
        } catch (e: Cancelled) {
            // Scan again or a new item: the new state is already on screen.
        } catch (e: Throwable) {
            if (gen == explainGeneration) {
                update { it.copy(explaining = false, message = "Llama failed: ${e.message}") }
            }
        } finally {
            llmBusy = false
        }
    }

    /** Load Llama the first time it is needed (on the work thread, under the model lock). */
    private fun ensureLlama(): Llama? {
        llama?.let { return it }
        val (model, tok) = llamaFiles ?: return null
        update { it.copy(llamaLoading = true) }
        setModelStatus("Llama", "loading ${model.name}…")
        return try {
            val t0 = SystemClock.elapsedRealtime()
            Llama(model, tok).also {
                it.load()
                llama = it
                setModelStatus("Llama", "ready: ${model.name} (load ${(SystemClock.elapsedRealtime() - t0) / 1000.0} s)")
            }
        } catch (e: Throwable) {
            llamaFiles = null
            setModelStatus("Llama", "error: ${e.message}")
            null
        } finally {
            update { it.copy(llamaLoading = false) }
        }
    }

    private fun setModelStatus(name: String, status: String) =
        update { it.copy(models = LinkedHashMap(it.models).apply { put(name, status) }) }

    /**
     * Settings > Prepare explanations: write every item's explanation on this phone now (each
     * item, and each condition answer for items that ask one), so later scans show it at once.
     * Pauses scanning while it runs; tapping again stops it.
     */
    fun prepareExplanations() {
        if (preparing) {
            preparing = false
            if (llmBusy) runCatching { llama?.stop() }
            return
        }
        val kb = kb ?: return
        if (llamaFiles == null && llama == null) {
            update { it.copy(message = "Llama is not on the phone") }
            return
        }
        preparing = true
        val language = ui.language
        val checker = checker
        val jobs = kb.itemIds.flatMap { id ->
            listOf(id to emptyMap<String, Any?>()) +
                checker?.attributesFor(id).orEmpty().flatMap { a -> listOf(id to mapOf<String, Any?>(a to true), id to mapOf<String, Any?>(a to false)) }
        }
        update { it.copy(preparing = "0 / ${jobs.size}") }
        work.execute {
            llmBusy = true
            try {
                val llama = ensureLlama() ?: return@execute
                fun gen(prompt: String, temperature: Float): String {
                    if (!preparing) throw Cancelled()
                    return llama.generate(prompt, temperature = temperature) { }
                }
                for ((i, job) in jobs.withIndex()) {
                    if (!preparing) break
                    val (id, attrs) = job
                    val key = ExplanationCache.key(kb.version, id, attrs, language)
                    if (explanations.get(key) == null) {
                        val verdict = kb.resolve(id, attrs, language)
                        val e = Assistant.explain(kb, verdict, attrs, { gen(it, 0f) }, null, sample = { gen(it, 0.7f) })
                        explanations.put(key, e.verified)
                    }
                    update { it.copy(preparing = "${i + 1} / ${jobs.size}", cachedAnswers = explanations.size()) }
                }
            } catch (e: Cancelled) {
                // stopped from Settings
            } catch (e: Throwable) {
                update { it.copy(message = "Preparing stopped: ${e.message}") }
            } finally {
                preparing = false
                llmBusy = false
                update { it.copy(preparing = null, cachedAnswers = explanations.size()) }
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
        locked = true  // stop scanning while the user chooses
        lastFrame = bitmap
        update { it.copy(lookingCloser = true, canSuggest = false, scanState = "unsure") }
        vision.execute {
            val emb = siglip.embed(bitmap).also { latency.record(Stages.FALLBACK, siglip.lastMs) }
            val suggestions = siglip.suggest(emb)
            update { it.copy(suggestions = suggestions, lookingCloser = false) }
        }
    }

    /** Shutter: decide on the current frame now instead of waiting for frame voting. */
    /** The camera's still-photo use case; bound by the screen next to preview and analysis. */
    @Volatile var imageCapture: ImageCapture? = null

    /**
     * Shutter: take a full-resolution photo and analyse it like an uploaded picture (detector,
     * agreement check, then a verdict or suggestions). Live scanning is untouched; without a still
     * camera (binding failed) the last analysis frame is used instead.
     */
    fun captureNow() {
        if (!ready) {
            update { it.copy(message = "Still starting up (about half a minute). Try again in a moment.") }
            return
        }
        if (capturing) return
        val capture = imageCapture
        if (capture == null) {
            lastFrame?.let(::photo)
            return
        }
        capturing = true
        locked = true  // stop live scanning while the photo is taken and analysed
        update { it.copy(lookingCloser = true, suggestions = emptyList(), canSuggest = false) }
        capture.takePicture(captureThread, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                // Copy the pixels now and give the buffer straight back to the camera.
                val bitmap = try {
                    image.toBitmap().rotate(image.imageInfo.rotationDegrees).limit(PHOTO_MAX_SIDE)
                } catch (e: IllegalStateException) {
                    null
                } finally {
                    image.close()
                }
                if (bitmap != null) photo(bitmap, fromShutter = true) else failed("the camera released the photo")
            }

            override fun onError(exception: ImageCaptureException) = failed(exception.message ?: "camera error")

            private fun failed(why: String) {
                capturing = false
                locked = false
                update { it.copy(lookingCloser = false, message = "Couldn't take a photo ($why). Try again.") }
            }
        })
    }

    /** Camera torch (flash); the camera itself is bound by the screen. */
    @Volatile var camera: androidx.camera.core.Camera? = null

    fun toggleTorch() {
        val cam = camera ?: return
        val on = !ui.torchOn
        cam.cameraControl.enableTorch(on)
        update { it.copy(torchOn = on) }
    }

    /** Plain text for the share sheet. */
    fun shareText(): String? {
        val v = ui.verdict ?: return null
        val hi = v.language == "hi"
        val head = if (hi) "${v.name}: ${v.binColourName} डिब्बा (${v.streamName})" else "${v.name}: ${v.binColourName} bin (${v.streamName})"
        val steps = v.steps.mapIndexed { i, step -> "${i + 1}. $step" }.joinToString("\n")
        return "$head\n$steps\n— SwachhSathi (offline, on-device)"
    }

    /** The bin colour of an item (for the suggestion list). */
    fun binColourOf(itemId: String): String? = kb?.resolve(itemId, emptyMap(), ui.language)?.binColour

    fun choose(itemId: String) {
        locked = true
        cancelExplanation()
        vision.execute { lock(itemId, lastFrame, null, SystemClock.elapsedRealtimeNanos()) }
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
        cancelExplanation()
        vision.execute { lock(itemId, frame, null, SystemClock.elapsedRealtimeNanos()) }
    }

    fun photo(bitmap: Bitmap, fromShutter: Boolean = false) {
        if (!ready) {
            if (fromShutter) capturing = false
            update { it.copy(message = "Still starting up (about half a minute). Try again in a moment.") }
            return
        }
        val det = detector
        lastFrame = bitmap
        cancelExplanation()
        locked = true
        vision.execute {
            try {
                val t0 = SystemClock.elapsedRealtimeNanos()
                val best = det?.detect(bitmap, latency)?.filter { it.itemId != null }?.maxByOrNull { it.confidence }
                if (best != null && best.confidence >= GATE) {
                    if (agrees(best.itemId!!, bitmap)) {
                        lock(best.itemId!!, bitmap.crop(best.bbox), best.confidence.toDouble(), t0)
                    }
                } else {
                    update { it.copy(verdict = null) }
                    suggest(bitmap)
                }
            } finally {
                if (fromShutter) capturing = false
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
            val all = scans.all()
            val summary = Impact.summary(kb, all, now, tz)
            val language = ui.language
            // Holding one item in view re-confirms it every few seconds: show it once, using the
            // same rule as the impact counts (same item again within DEDUPE_SECONDS).
            val distinct = mutableListOf<Scan>()
            for (scan in all) {
                val last = distinct.lastOrNull()
                if (last != null && last.itemId == scan.itemId && scan.timeS - last.timeS <= Impact.DEDUPE_SECONDS) {
                    distinct[distinct.size - 1] = scan
                } else distinct.add(scan)
            }
            val history = distinct.asReversed().take(100).mapNotNull { scan ->
                runCatching {
                    val v = kb.resolve(scan.itemId, scan.attributes, language)
                    HistoryRow((scan.timeS * 1000).toLong(), scan.itemId, v.name, v.binColour, v.binColourName, v.streamName)
                }.getOrNull()
            }
            update {
                it.copy(impact = summary, didYouKnow = Education.didYouKnow(kb, day, it.language),
                    quiz = Education.quiz(kb, day, it.language), history = history)
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
        vision.shutdown()
        captureThread.shutdown()
        voiceThread.shutdown()
    }

    companion object {
        const val GATE = 0.5f
        /** No stable item for this long: show the "Suggest" button. */
        const val SUGGEST_BUTTON_MS = 1000L
        /** The detector sees something but stays unsure this long: suggest without a tap. */
        const val AUTO_SUGGEST_MS = 1500L
        /** Llama starts this long after a card appears (not for typed or spoken questions). */
        const val EXPLAIN_DELAY_MS = 1500L
        /** Shutter photos are scaled down to this long side: plenty for 640 px models, less memory. */
        const val PHOTO_MAX_SIDE = 1600

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

/** Scale down so the longer side is at most [maxSide] pixels (never up). */
private fun Bitmap.limit(maxSide: Int): Bitmap {
    val side = maxOf(width, height)
    if (side <= maxSide) return this
    val f = maxSide.toFloat() / side
    return Bitmap.createScaledBitmap(this, (width * f).toInt(), (height * f).toInt(), true)
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
