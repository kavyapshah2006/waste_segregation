package wastesegregation.core

import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertEquals

/** tests/fixtures/detect_cases.json, the same cases as tests/test_detect_cases.py. */
@Suppress("UNCHECKED_CAST")
class DetectTest {
    private val data = TestKb.readJson("tests/fixtures/detect_cases.json") as Map<String, Any?>

    private fun num(v: Any?) = (v as JsonNumber).text.toDouble()

    @TestFactory
    fun letterbox() = (data["letterbox"] as List<Map<String, Any?>>).map { case ->
        val w = num(case["w"]).toInt()
        val h = num(case["h"]).toInt()
        DynamicTest.dynamicTest("letterbox ${w}x$h") {
            val g = Detect.letterboxGeometry(w, h, num(case["size"]).toInt())
            val actual = linkedMapOf<String, Any?>(
                "scale" to g.letterbox.scale, "new_w" to g.newW, "new_h" to g.newH,
                "left" to g.letterbox.left, "top" to g.letterbox.top,
            )
            assertEquals(case["expected"], Json.parse(Json.dumps(actual)))
        }
    }

    @TestFactory
    fun decode() = (data["decode"] as List<Map<String, Any?>>).map { case ->
        DynamicTest.dynamicTest(case["id"] as String) {
            val image = (case["image"] as List<Any?>).map { num(it).toInt() }
            val lb = Detect.letterboxGeometry(image[0], image[1]).letterbox
            // Like numpy: decimal text -> float64 -> float32.
            val output = (case["output"] as List<List<Any?>>)
                .map { row -> row.map { num(it).toFloat() }.toFloatArray() }
                .toTypedArray()
            val dets = Detect.decode(
                output, lb, case["names"] as List<String>, num(case["conf"]), num(case["iou"]),
                num(case["max_det"]).toInt(), (case["known"] as List<String>).toSet(),
            )
            assertEquals(case["expected"], Json.parse(Json.dumps(dets.map { it.toMap() })))
        }
    }
}
