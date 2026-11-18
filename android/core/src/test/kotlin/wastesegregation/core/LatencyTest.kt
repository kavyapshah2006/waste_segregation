package wastesegregation.core

import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/** tests/fixtures/latency_cases.json, the same cases as tests/test_latency_stages.py. */
@Suppress("UNCHECKED_CAST")
class LatencyTest {
    private val cases = TestKb.readJson("tests/fixtures/latency_cases.json") as List<Map<String, Any?>>

    private fun num(v: Any?) = (v as JsonNumber).text.toDouble()

    @TestFactory
    fun fixtures() = cases.map { case ->
        DynamicTest.dynamicTest(case["id"] as String) {
            val samples = (case["samples"] as Map<String, List<Any?>>).mapValues { e -> e.value.map(::num) }
            val summary = LatencyLog.summarise(samples)
            // Compare as numbers: the fixture writes 4 as 4.0, Kotlin keeps n as an Int.
            val expected = (case["expected"] as Map<String, Map<String, Any?>>)
                .mapValues { e -> e.value.mapValues { num(it.value) } }
            val actual = summary.mapValues { e -> e.value.mapValues { (it.value as Number).toDouble() } }
            assertEquals(expected, actual)
            (case["expected_order"] as List<String>?)?.let { assertEquals(it, summary.keys.toList()) }
        }
    }

    @Test
    fun timesWithInjectedClock() {
        var now = 0L
        val log = LatencyLog(clock = { now })
        val result = log.time(Stages.DETECT) { now += 12_500_000L; "done" }
        assertEquals("done", result)
        assertEquals(listOf(12.5), log.samples(Stages.DETECT))
    }

    @Test
    fun recordsEvenWhenTheStageThrows() {
        var now = 0L
        val log = LatencyLog(clock = { now })
        assertFailsWith<IllegalStateException> {
            log.time(Stages.KB) { now += 1_000_000L; error("boom") }
        }
        assertEquals(listOf(1.0), log.samples(Stages.KB))
    }

    @Test
    fun exportRoundTrips() {
        val log = LatencyLog(clock = { 0L })
        log.record(Stages.DETECT, 10.0)
        log.record(Stages.DETECT, 12.0)
        val parsed = Json.parse(log.toJson(mapOf("commit" to "abc1234"))) as Map<String, Any?>
        assertEquals("stage_latency", parsed["kind"])
        assertEquals("abc1234", parsed["commit"])
        val detect = (parsed["samples"] as Map<String, List<Any?>>).getValue("detect").map(::num)
        assertEquals(listOf(10.0, 12.0), detect)
    }

    @Test
    fun exportRoundsTinyDurations() {
        val log = LatencyLog(clock = { 0L })
        log.record(Stages.KB, 0.0008)
        log.record(Stages.KB, 0.0004)
        log.record(Stages.KB, 3.14159)
        val parsed = Json.parse(log.toJson(emptyMap())) as Map<String, Any?>
        val kb = (parsed["samples"] as Map<String, List<Any?>>).getValue("kb").map(::num)
        assertEquals(listOf(0.001, 0.0, 3.142), kb)
    }

    @Test
    fun rejectsNegativeDurations() {
        assertFailsWith<IllegalArgumentException> { LatencyLog().record(Stages.KB, -1.0) }
    }
}
