package wastesegregation.core

import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.TestFactory
import kotlin.test.Test
import kotlin.test.assertEquals

/** tests/fixtures/impact/cases.json, the same cases as tests/test_impact.py. */
@Suppress("UNCHECKED_CAST")
class ImpactTest {
    private val kb = TestKb.kb
    private val cases = TestKb.readJson("tests/fixtures/impact/cases.json") as List<Map<String, Any?>>

    private fun num(v: Any?) = (v as JsonNumber).text.toDouble()

    private fun compute(case: Map<String, Any?>): Any? {
        val input = case["input"] as Map<String, Any?>
        return when (case["kind"]) {
            "summary" -> Impact.summary(
                kb,
                (input["scans"] as List<Map<String, Any?>>).map {
                    Scan(num(it["time_s"]), it["item_id"] as String, it["attributes"] as Map<String, Any?>? ?: emptyMap())
                },
                num(input["now_s"]),
                (input["tz_offset_minutes"] as JsonNumber?)?.text?.toInt() ?: 330,
            )
            "did_you_know" -> Education.didYouKnow(kb, num(input["day"]).toLong(), input["language"] as String)
            "quiz" -> Education.quiz(kb, num(input["day"]).toLong(), input["language"] as String)
            "watch_out" -> Education.watchOutCards(kb, input["language"] as String)
            else -> error("unknown kind ${case["kind"]}")
        }
    }

    @TestFactory
    fun fixtures() = cases.map { case ->
        DynamicTest.dynamicTest(case["id"] as String) {
            // Round-trip through JSON so numbers compare the way the fixture stores them.
            assertEquals(case["expected"], Json.parse(Json.dumps(compute(case))))
        }
    }

    @Test
    fun `round3 matches python round`() {
        assertEquals(0.167, Impact.round3(1.0 / 6))
        assertEquals(2.675, Impact.round3(2.675)) // 2.67499999... in binary
        assertEquals(0.0, Impact.round3(0.0004))
    }
}
