package wastesegregation.core

import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertEquals

/** tests/fixtures/fallback_rank.json, the same cases as tests/test_fallback.py. */
@Suppress("UNCHECKED_CAST")
class FallbackRankTest {
    private val data = TestKb.readJson("tests/fixtures/fallback_rank.json") as Map<String, Any?>

    private fun num(v: Any?) = (v as JsonNumber).text.toDouble()

    private fun floats(v: Any?) = (v as List<Any?>).map { num(it).toFloat() }.toFloatArray()

    @TestFactory
    fun fixtures() = (data["cases"] as List<Map<String, Any?>>).mapIndexed { i, case ->
        DynamicTest.dynamicTest("case $i") {
            val result = FallbackRank.rank(
                (data["text"] as List<Any?>).map { floats(it) },
                data["item_ids"] as List<String>,
                floats(case["image"]),
                num(data["logit_scale"]),
                num(data["logit_bias"]),
                num(case["k"]).toInt(),
            )
            assertEquals(case["expected"], Json.parse(Json.dumps(result.map { it.toMap() })))
        }
    }
}
