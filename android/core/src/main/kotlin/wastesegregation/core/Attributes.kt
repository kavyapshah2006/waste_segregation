package wastesegregation.core

import java.math.BigDecimal
import java.math.RoundingMode

/** One attribute result: [value] is null when the app should ask the user instead of guessing. */
data class AttributeResult(val value: Boolean?, val margin: Double) {
    fun toMap(): Map<String, Any?> = linkedMapOf("value" to value, "margin" to margin)
}

/**
 * Dirty / broken check from the SigLIP 2 image embedding: a port of pipeline/attributes.py
 * (shared cases: tests/fixtures/attribute_cases.json). Built from attribute_prompts.json
 * (models/attributes/build.py): per attribute a {low, high} band, per item and attribute a
 * positive and a negative text embedding.
 */
class AttributeChecker(prompts: Map<String, Any?>) {
    private val thresholds: Map<String, Pair<Double, Double>>
    private val items: Map<String, Map<String, Pair<DoubleArray, DoubleArray>>>

    init {
        @Suppress("UNCHECKED_CAST")
        val t = prompts["thresholds"] as Map<String, Map<String, Any?>>
        thresholds = t.mapValues { (_, v) -> num(v["low"]) to num(v["high"]) }
        @Suppress("UNCHECKED_CAST")
        val i = prompts["items"] as Map<String, Map<String, Map<String, List<Any?>>>>
        items = i.mapValues { (_, attrs) ->
            attrs.mapValues { (_, spec) -> vector(spec.getValue("positive")) to vector(spec.getValue("negative")) }
        }
    }

    fun attributesFor(itemId: String): List<String> = items[itemId]?.keys?.sorted() ?: emptyList()

    fun check(itemId: String, imageEmbedding: FloatArray): Map<String, AttributeResult> {
        val out = linkedMapOf<String, AttributeResult>()
        for (attribute in attributesFor(itemId)) {
            val (positive, negative) = items.getValue(itemId).getValue(attribute)
            val m = margin(imageEmbedding, positive, negative)
            val (low, high) = thresholds.getValue(attribute)
            out[attribute] = AttributeResult(decide(m, low, high), round4(m))
        }
        return out
    }

    companion object {
        fun margin(image: FloatArray, positive: DoubleArray, negative: DoubleArray): Double {
            require(image.size == positive.size && image.size == negative.size) { "embedding size mismatch" }
            var pos = 0.0
            var neg = 0.0
            for (k in image.indices) {
                pos += image[k].toDouble() * positive[k]
                neg += image[k].toDouble() * negative[k]
            }
            return pos - neg
        }

        fun decide(value: Double, low: Double, high: Double): Boolean? {
            require(low <= high) { "low $low > high $high" }
            return when {
                value > high -> true
                value < low -> false
                else -> null
            }
        }

        private fun num(v: Any?): Double = when (v) {
            is JsonNumber -> v.text.toDouble()
            is Number -> v.toDouble()
            else -> error("not a number: $v")
        }

        private fun vector(values: List<Any?>) = DoubleArray(values.size) { num(values[it]) }

        private fun round4(x: Double) = BigDecimal(x).setScale(4, RoundingMode.HALF_EVEN).toDouble()
    }
}
