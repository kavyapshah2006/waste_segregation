package wastesegregation.core

import java.math.BigDecimal
import java.math.RoundingMode
import kotlin.math.exp

/**
 * Ranking for the unknown-item fallback: a port of rank() in pipeline/fallback.py. The app
 * runs the SigLIP 2 image encoder (.pte) itself; this turns its embedding into the top-k KB
 * items to offer the user (suggestions to confirm, never a verdict).
 *
 * Python keeps embeddings as float32 and takes the dot product in float64; values are
 * rounded through Float here the same way so both sides agree to the last digit.
 */
data class Suggestion(val itemId: String, val cosine: Double, val score: Double) {
    fun toMap(): Map<String, Any?> = linkedMapOf("item_id" to itemId, "cosine" to cosine, "score" to score)
}

object FallbackRank {
    fun rank(
        text: List<FloatArray>,
        itemIds: List<String>,
        imageEmbedding: FloatArray,
        logitScale: Double,
        logitBias: Double,
        k: Int = 3,
    ): List<Suggestion> {
        val cos = text.map { row ->
            var s = 0.0
            for (i in row.indices) s += row[i].toDouble() * imageEmbedding[i].toDouble()
            s
        }
        return cos.indices
            .sortedWith(compareBy({ -cos[it] }, { it }))
            .take(k)
            .map { i ->
                val score = 1 / (1 + exp(-(logitScale * cos[i] + logitBias)))
                Suggestion(itemIds[i], round4(cos[i]), round4(score))
            }
    }

    private fun round4(x: Double) = BigDecimal(x).setScale(4, RoundingMode.HALF_EVEN).toDouble()
}
