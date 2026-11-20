package wastesegregation.core

/**
 * Decides when the speaker has finished, from the loudness of each chunk. The quietest of the
 * first chunks is taken as the room's noise; speech is anything clearly above it.
 */
class Endpointer(private val silenceMs: Int, private val noSpeechMs: Int) {
    private var elapsedMs = 0
    private var noise = Float.MAX_VALUE
    private var spoke = false
    private var quietMs = 0

    /** Feed one chunk's RMS loudness and length; true once recording should stop. */
    fun done(rms: Float, ms: Int): Boolean {
        elapsedMs += ms
        if (elapsedMs <= CALIBRATE_MS) noise = minOf(noise, rms)
        val threshold = maxOf(MIN_SPEECH_RMS, noise * NOISE_FACTOR)
        if (rms >= threshold) {
            spoke = true
            quietMs = 0
        } else if (spoke) {
            quietMs += ms
        }
        return if (spoke) quietMs >= silenceMs else elapsedMs >= noSpeechMs
    }

    companion object {
        const val CALIBRATE_MS = 300
        const val NOISE_FACTOR = 3f
        const val MIN_SPEECH_RMS = 0.015f
    }
}
