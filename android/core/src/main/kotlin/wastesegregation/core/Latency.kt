package wastesegregation.core

import kotlin.math.ceil
import kotlin.math.max

/** Stage names for the latency harness, in pipeline order. Same strings as eval/latency/stages.py. */
object Stages {
    const val CAPTURE = "capture"
    const val PREPROCESS = "preprocess"
    const val DETECT = "detect"
    const val ATTRIBUTES = "attributes"
    const val VOTE = "vote"
    const val KB = "kb"
    const val FRAME_TO_CARD = "frame_to_card"
    const val FALLBACK = "fallback"
    const val LLM_FIRST_TOKEN = "llm_first_token"
    const val ASR = "asr"
    const val TTS_START = "tts_start"

    val ORDER = listOf(
        CAPTURE, PREPROCESS, DETECT, ATTRIBUTES, VOTE, KB, FRAME_TO_CARD, FALLBACK,
        LLM_FIRST_TOKEN, ASR, TTS_START,
    )
}

/**
 * Per-stage timings for one app session (FuturePlan WP11, step 60), exported as
 * eval/results/latency_<device>_<date>.json and summarised by eval/latency/stages.py.
 * Durations are in milliseconds. The clock is injectable for tests; on the phone it is
 * System.nanoTime.
 */
class LatencyLog(val clock: () -> Long = System::nanoTime) {
    private val samples = linkedMapOf<String, MutableList<Double>>()

    fun record(stage: String, ms: Double) {
        require(ms >= 0.0 && !ms.isNaN()) { "bad duration for $stage: $ms" }
        samples.getOrPut(stage) { mutableListOf() }.add(ms)
    }

    /** Runs [block] and records how long it took under [stage]. */
    inline fun <T> time(stage: String, block: () -> T): T {
        val start = clock()
        try {
            return block()
        } finally {
            record(stage, (clock() - start) / 1_000_000.0)
        }
    }

    fun samples(stage: String): List<Double> = samples[stage]?.toList() ?: emptyList()

    fun summary(): Map<String, Map<String, Any>> = summarise(samples)

    /** The export file: [meta] holds device, commit and conditions (see the template). */
    fun toJson(meta: Map<String, Any?>): String {
        val out = linkedMapOf<String, Any?>("kind" to "stage_latency")
        out.putAll(meta)
        out["samples"] = samples.mapValues { it.value.toList() }
        return Json.dumps(out)
    }

    companion object {
        /** Nearest-rank percentile, as in eval/latency/stages.py. */
        fun percentile(values: List<Double>, q: Double): Double? {
            if (values.isEmpty()) return null
            require(q > 0.0 && q <= 1.0) { "q must be in (0, 1], got $q" }
            val ordered = values.sorted()
            val rank = max(1, ceil(q * ordered.size).toInt())
            return ordered[rank - 1]
        }

        /** {stage: {n, p50, p95, max}}: known stages in pipeline order, then others sorted. */
        fun summarise(samples: Map<String, List<Double>>): Map<String, Map<String, Any>> {
            val known = Stages.ORDER.filter { it in samples }
            val extra = samples.keys.filter { it !in Stages.ORDER }.sorted()
            val out = linkedMapOf<String, Map<String, Any>>()
            for (stage in known + extra) {
                val values = samples.getValue(stage)
                if (values.isEmpty()) continue
                require(values.all { it >= 0.0 }) { "negative duration in $stage" }
                out[stage] = linkedMapOf(
                    "n" to values.size,
                    "p50" to percentile(values, 0.5)!!,
                    "p95" to percentile(values, 0.95)!!,
                    "max" to values.max(),
                )
            }
            return out
        }
    }
}
