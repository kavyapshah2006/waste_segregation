package wastesegregation.core

import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertEquals

/** tests/fixtures/stability_cases.json, the same cases as tests/test_stability.py. */
@Suppress("UNCHECKED_CAST")
class StabilityTest {
    private val cases = TestKb.readJson("tests/fixtures/stability_cases.json") as List<Map<String, Any?>>

    private fun num(v: Any?) = (v as JsonNumber).text.toDouble()

    @TestFactory
    fun fixtures() = cases.map { case ->
        DynamicTest.dynamicTest(case["id"] as String) {
            val p = case["params"] as Map<String, Any?>
            val voter = FrameVoter(num(p["window"]).toInt(), num(p["min_votes"]).toInt(), num(p["gate"]))
            val states = (case["frames"] as List<List<Map<String, Any?>>>).map { frame ->
                voter.push(frame.map { Detected(it["item_id"] as String?, num(it["confidence"])) }).toMap()
            }
            assertEquals(case["expected"], Json.parse(Json.dumps(states)))
        }
    }
}
