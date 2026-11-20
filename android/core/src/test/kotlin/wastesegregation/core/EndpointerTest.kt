package wastesegregation.core

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Test

class EndpointerTest {
    /** Feeds 100 ms chunks; returns the time (ms) at which the endpointer said stop, or null. */
    private fun stopAt(levels: List<Float>, silenceMs: Int = 1200, noSpeechMs: Int = 5000): Int? {
        val e = Endpointer(silenceMs, noSpeechMs)
        levels.forEachIndexed { i, rms -> if (e.done(rms, 100)) return (i + 1) * 100 }
        return null
    }

    private fun quiet(ms: Int) = List(ms / 100) { 0.003f }
    private fun speech(ms: Int) = List(ms / 100) { 0.08f }

    @Test
    fun `stops a little after the speaker finishes`() {
        // 0.5 s quiet, 2 s of speech, then quiet: stop 1.2 s after the speech ends.
        assertEquals(3700, stopAt(quiet(500) + speech(2000) + quiet(3000)))
    }

    @Test
    fun `a short pause inside a sentence does not stop it`() {
        val levels = quiet(500) + speech(1000) + quiet(700) + speech(1000) + quiet(3000)
        assertEquals(4400, stopAt(levels))
    }

    @Test
    fun `nothing said stops after the no-speech limit`() {
        assertEquals(5000, stopAt(quiet(8000)))
    }

    @Test
    fun `a noisy room raises the speech threshold`() {
        // Fan noise at 0.03 is not speech; speech at 0.12 is.
        val noisy = List(60) { 0.03f }
        assertEquals(5000, stopAt(noisy))
        val levels = List(5) { 0.03f } + List(10) { 0.12f } + List(20) { 0.03f }
        assertEquals(2700, stopAt(levels))
    }

    @Test
    fun `speaking at once still records everything`() {
        // Speech from the first chunk makes the noise estimate high, so no early stop is found;
        // the recording simply runs to its limit and nothing is cut off.
        assertNull(stopAt(speech(3000)))
    }
}
