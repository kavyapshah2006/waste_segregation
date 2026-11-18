package wastesegregation.core

import java.math.BigDecimal
import java.math.RoundingMode
import kotlin.math.floor

/**
 * Detector post-processing: a port of pipeline/detect.py (letterbox geometry, decode, NMS,
 * mapping back to the image). The app does the pixel resize itself (Android's bitmap scaling
 * cannot match PIL bit for bit); the geometry and everything after the model are exact.
 *
 * Arithmetic is in Float to mirror NumPy float32: Python numbers mixed into float32 arrays
 * stay float32 (NEP 50), so thresholds and the letterbox scale are cast to Float here too.
 */
data class Letterbox(val scale: Double, val left: Int, val top: Int, val width: Int, val height: Int)

data class LetterboxSize(val letterbox: Letterbox, val newW: Int, val newH: Int)

data class Detection(
    val itemId: String?,
    val detectorClass: String,
    val confidence: Float,
    val bbox: List<Float>,
) {
    /** Same keys and rounding as Detection.to_dict() in Python. */
    fun toMap(): Map<String, Any?> = linkedMapOf(
        "item_id" to itemId,
        "detector_class" to detectorClass,
        "confidence" to round4(confidence),
        "bbox" to bbox.map { round4(it) },
        "attributes" to emptyMap<String, Any?>(),
        "source" to "detector",
    )

    private fun round4(x: Float) = BigDecimal(x.toDouble()).setScale(4, RoundingMode.HALF_EVEN).toDouble()
}

object Detect {
    const val INPUT_SIZE = 640
    const val PAD_VALUE = 114
    const val CONF_THRESHOLD = 0.25
    const val IOU_THRESHOLD = 0.7
    const val MAX_DETECTIONS = 100

    fun letterboxGeometry(w: Int, h: Int, size: Int = INPUT_SIZE): LetterboxSize {
        val r = minOf(size.toDouble() / h, size.toDouble() / w)
        val newW = Math.rint(w * r).toInt() // half to even, like Python's round()
        val newH = Math.rint(h * r).toInt()
        val left = floor((size - newW) / 2.0).toInt()
        val top = floor((size - newH) / 2.0).toInt()
        return LetterboxSize(Letterbox(r, left, top, w, h), newW, newH)
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val inter = maxOf(minOf(a[2], b[2]) - maxOf(a[0], b[0]), 0f) *
            maxOf(minOf(a[3], b[3]) - maxOf(a[1], b[1]), 0f)
        val areaA = (a[2] - a[0]) * (a[3] - a[1])
        val areaB = (b[2] - b[0]) * (b[3] - b[1])
        return inter / maxOf(areaA + areaB - inter, 1e-9f)
    }

    /** Class-aware NMS; indices into the inputs, highest score first (ties: lower index). */
    fun nms(boxes: List<FloatArray>, scores: FloatArray, classes: IntArray, iou: Double, maxDet: Int): List<Int> {
        var order = scores.indices.sortedWith(compareBy({ -scores[it] }, { it }))
        val keep = mutableListOf<Int>()
        val threshold = iou.toFloat()
        while (order.isNotEmpty() && keep.size < maxDet) {
            val i = order[0]
            keep.add(i)
            order = order.drop(1).filterNot { classes[it] == classes[i] && iou(boxes[i], boxes[it]) > threshold }
        }
        return keep
    }

    /**
     * [output] is the model output laid out as (4 + classes) rows x anchors columns: centre x,
     * centre y, width, height in input pixels, then one score per class.
     */
    fun decode(
        output: Array<FloatArray>,
        lb: Letterbox,
        classNames: List<String>,
        conf: Double = CONF_THRESHOLD,
        iou: Double = IOU_THRESHOLD,
        maxDet: Int = MAX_DETECTIONS,
        knownItems: Set<String> = emptySet(),
    ): List<Detection> {
        require(output.size == 4 + classNames.size) {
            "model has ${output.size - 4} classes, names list has ${classNames.size}"
        }
        val anchors = output[0].size
        val confF = conf.toFloat()
        val boxes = mutableListOf<FloatArray>()
        val scores = mutableListOf<Float>()
        val classes = mutableListOf<Int>()
        for (a in 0 until anchors) {
            var best = 0
            for (c in 1 until classNames.size) if (output[4 + c][a] > output[4 + best][a]) best = c
            val score = output[4 + best][a]
            if (score < confF) continue
            val cx = output[0][a]
            val cy = output[1][a]
            val w = output[2][a]
            val h = output[3][a]
            boxes.add(floatArrayOf(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            scores.add(score)
            classes.add(best)
        }
        if (boxes.isEmpty()) return emptyList()
        val scale = lb.scale.toFloat()
        fun mapX(x: Float) = minOf(maxOf((x - lb.left) / scale, 0f), lb.width.toFloat()) / lb.width
        fun mapY(y: Float) = minOf(maxOf((y - lb.top) / scale, 0f), lb.height.toFloat()) / lb.height
        return nms(boxes, scores.toFloatArray(), classes.toIntArray(), iou, maxDet).map { i ->
            val b = boxes[i]
            val name = classNames[classes[i]]
            Detection(
                itemId = if (name in knownItems) name else null,
                detectorClass = name,
                confidence = scores[i],
                bbox = listOf(mapX(b[0]), mapY(b[1]), mapX(b[2]), mapY(b[3])),
            )
        }
    }
}
