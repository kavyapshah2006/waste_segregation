package org.wastesegregation.app

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineOmnilingualAsrCtcModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import java.io.File

/**
 * Push-to-talk speech recognition, fully offline: Meta Omnilingual ASR 300M CTC v2 (int8,
 * Apache-2.0) through sherpa-onnx. Chosen on the laptop (models/speech/README.md); the
 * transcript is matched to KB items by pronunciation in the core's Voice.match.
 *
 * The 366 MB model is loaded for each utterance and released right after, so it never sits in
 * memory next to Llama (FuturePlan WP9 step 5).
 */
class SpeechInput(private val model: File, private val tokens: File) {
    @Volatile private var recording = false

    /** Records 16 kHz mono until [stop] is called or [maxSeconds] pass. Blocking. */
    @SuppressLint("MissingPermission") // the caller checks RECORD_AUDIO first
    fun record(maxSeconds: Int = 8): FloatArray {
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, SAMPLE_RATE / 5 * 2),
        )
        val samples = FloatArray(SAMPLE_RATE * maxSeconds)
        val chunk = ShortArray(SAMPLE_RATE / 10)
        var n = 0
        recording = true
        recorder.startRecording()
        try {
            while (recording && n < samples.size) {
                val read = recorder.read(chunk, 0, minOf(chunk.size, samples.size - n))
                if (read <= 0) break
                for (i in 0 until read) samples[n + i] = chunk[i] / 32768f
                n += read
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

    /** Transcribes [samples] (16 kHz mono, -1..1). Loads and releases the model each time. */
    fun transcribe(samples: FloatArray, threads: Int = 4): String {
        val config = OfflineRecognizerConfig(
            modelConfig = OfflineModelConfig(
                omnilingual = OfflineOmnilingualAsrCtcModelConfig(model = model.path),
                tokens = tokens.path,
                numThreads = threads,
            ),
        )
        val recognizer = OfflineRecognizer(config = config)
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
