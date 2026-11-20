package org.wastesegregation.app

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineOmnilingualAsrCtcModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import wastesegregation.core.Endpointer
import java.io.File
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.Future

/**
 * Push-to-talk speech recognition, fully offline: Meta Omnilingual ASR 300M CTC v2 (int8,
 * Apache-2.0) through sherpa-onnx. Chosen on the laptop (models/speech/README.md); the
 * transcript is matched to KB items by pronunciation in the core's Voice.match.
 *
 * The 366 MB model is loaded for each utterance and released right after, so it does not stay in
 * memory between questions (FuturePlan WP9 step 5). It starts loading as soon as recording
 * starts ([preload]), so the load overlaps the user speaking instead of following it.
 *
 * Recording stops by itself about a second after the user stops talking: on the phone, users who
 * did not tap Stop waited the full 8 s with no sign anything was happening.
 */
class SpeechInput(private val model: File, private val tokens: File, private val threads: Int = 4) {
    @Volatile private var recording = false
    private val loader = Executors.newSingleThreadExecutor()
    @Volatile private var loading: Future<OfflineRecognizer>? = null

    /**
     * Records 16 kHz mono until [stop] is called, the speaker has been quiet for [silenceMs]
     * after speaking, nothing was said for [noSpeechMs], or [maxSeconds] pass. [onLevel] gets the
     * loudness of each 100 ms (0..1) for a level meter. Blocking.
     */
    @SuppressLint("MissingPermission") // the caller checks RECORD_AUDIO first
    fun record(
        maxSeconds: Int = 8,
        silenceMs: Int = 1200,
        noSpeechMs: Int = 5000,
        onLevel: (Float) -> Unit = {},
    ): FloatArray {
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, SAMPLE_RATE / 5 * 2),
        )
        val samples = FloatArray(SAMPLE_RATE * maxSeconds)
        val chunk = ShortArray(SAMPLE_RATE / 10)
        var n = 0
        val endpoint = Endpointer(silenceMs, noSpeechMs)
        recording = true
        recorder.startRecording()
        try {
            while (recording && n < samples.size) {
                val read = recorder.read(chunk, 0, minOf(chunk.size, samples.size - n))
                if (read <= 0) break
                var sum = 0.0
                for (i in 0 until read) {
                    val s = chunk[i] / 32768f
                    samples[n + i] = s
                    sum += s * s
                }
                n += read
                val rms = Math.sqrt(sum / read).toFloat()
                onLevel(rms)
                if (endpoint.done(rms, read * 1000 / SAMPLE_RATE)) break
            }
        } finally {
            recording = false
            recorder.stop()
            recorder.release()
        }
        return samples.copyOf(n)
    }

    fun stop() {
        recording = false
    }

    private fun newRecognizer(): OfflineRecognizer = OfflineRecognizer(
        config = OfflineRecognizerConfig(
            modelConfig = OfflineModelConfig(
                omnilingual = OfflineOmnilingualAsrCtcModelConfig(model = model.path),
                tokens = tokens.path,
                numThreads = threads,
            ),
        ),
    )

    /** Start loading the model in the background (call when recording starts). */
    @Synchronized
    fun preload() {
        if (loading == null) loading = loader.submit(Callable { newRecognizer() })
    }

    /** Free a preloaded model that will not be used (nothing was said). */
    @Synchronized
    fun discard() {
        val pending = loading ?: return
        loading = null
        loader.execute { runCatching { pending.get().release() } }
    }

    @Synchronized
    private fun takeRecognizer(): Future<OfflineRecognizer> =
        (loading ?: loader.submit(Callable { newRecognizer() })).also { loading = null }

    /** Transcribes [samples] (16 kHz mono, -1..1). Uses the preloaded model, releases it after. */
    fun transcribe(samples: FloatArray): String {
        val recognizer = takeRecognizer().get()
        try {
            val stream = recognizer.createStream()
            try {
                stream.acceptWaveform(samples, SAMPLE_RATE)
                recognizer.decode(stream)
                return recognizer.getResult(stream).text.trim()
            } finally {
                stream.release()
            }
        } finally {
            recognizer.release()
        }
    }

    companion object {
        const val SAMPLE_RATE = 16000
    }
}
