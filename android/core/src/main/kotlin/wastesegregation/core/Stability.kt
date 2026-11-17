package wastesegregation.core

import java.math.BigDecimal
import java.math.RoundingMode

/**
 * Confidence gate and voting over camera frames: a port of pipeline/stability.py. The app
 * shows a verdict only once the same item has won min_votes of the last window frames.
 */
data class FrameVote(val itemId: String?, val confidence: Double, val detected: Boolean)

data class Detected(val itemId: String?, val confidence: Double)

data class StabilityState(val state: String, val itemId: String?, val votes: Int, val confidence: Double?) {
    fun toMap(): Map<String, Any?> = linkedMapOf(
        "state" to state,
        "item_id" to itemId,
        "votes" to votes,
        "confidence" to confidence,
    )
}

class FrameVoter(
    private val window: Int = 5,
    private val minVotes: Int = 3,
    private val gate: Double = 0.5,
) {
    private val votes = ArrayDeque<FrameVote>()

    init {
        require(minVotes in 1..window) { "need 1 <= minVotes <= window" }
    }

    fun reset() = votes.clear()

    fun push(detections: List<Detected>): StabilityState {
        votes.addLast(frameVote(detections, gate))
        if (votes.size > window) votes.removeFirst()
        return state()
    }

    fun state(): StabilityState {
        val counts = linkedMapOf<String, Int>()
        for (v in votes) v.itemId?.let { counts[it] = (counts[it] ?: 0) + 1 }
        if (counts.isNotEmpty()) {
            val meanConf = counts.mapValues { (item, n) ->
                votes.filter { it.itemId == item }.sumOf { it.confidence } / n
            }
            val item = counts.keys.minWith(
                compareBy<String>({ -counts.getValue(it) }, { -meanConf.getValue(it) }, { it }),
            )
            if (counts.getValue(item) >= minVotes) {
                return StabilityState("stable", item, counts.getValue(item), round4(meanConf.getValue(item)))
            }
        }
        if (votes.any { it.detected }) return StabilityState("unsure", null, 0, null)
        return StabilityState("empty", null, 0, null)
    }

    companion object {
        fun frameVote(detections: List<Detected>, gate: Double = 0.5): FrameVote {
            val items = detections.filter { !it.itemId.isNullOrEmpty() }
            if (items.isEmpty()) return FrameVote(null, 0.0, false)
            val best = items.maxWith(compareBy<Detected>({ it.confidence }, { it.itemId }))
            return FrameVote(if (best.confidence >= gate) best.itemId else null, best.confidence, true)
        }

        /** Python's round(x, 4). */
        private fun round4(x: Double) = BigDecimal(x).setScale(4, RoundingMode.HALF_EVEN).toDouble()
    }
}
