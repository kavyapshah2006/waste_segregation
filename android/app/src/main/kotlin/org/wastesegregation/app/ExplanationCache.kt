package org.wastesegregation.app

import wastesegregation.core.Json
import wastesegregation.core.VerifiedOutput
import java.io.File

/**
 * Explanations Llama already wrote on this phone, so a later scan of the same item shows its
 * "why" at once instead of waiting ~15 s for Llama to read the prompt again.
 *
 * Only plain scans are cached (no typed or spoken question), and only answers that passed the
 * KB checks ("ok" or "corrected", or "partial": the sentences written before "Scan again" stopped
 * Llama, finished later); a fallback is retried next time. The key holds everything the
 * prompt depends on: KB version, item, condition answers and language. Generation is greedy, so
 * the first answer is the one Llama would give again; sampled retries only run when the greedy
 * answer fails the checks. Everything stays on the phone (app-private storage).
 */
class ExplanationCache(private val file: File) {
    private val entries = LinkedHashMap<String, VerifiedOutput>()

    init {
        if (file.exists()) {
            file.readLines().filter { it.isNotBlank() }.forEach { line ->
                runCatching {
                    val (key, value) = parse(line)
                    entries[key] = value
                }
            }
        }
    }

    @Synchronized
    fun get(key: String): VerifiedOutput? = entries[key]

    /**
     * Add the explanations shipped in the APK (written ahead with the same model, prompt and
     * checks; models/llm/seed_explanations.py) for keys this phone has no full answer for.
     * Kept in memory only: the APK is read again at each start. Returns how many were added.
     */
    @Synchronized
    fun seed(file: File): Int {
        var added = 0
        file.readLines().filter { it.isNotBlank() }.forEach { line ->
            runCatching {
                val (key, value) = parse(line)
                val old = entries[key]
                if (value.status in SAVED && (old == null || old.status == "partial")) {
                    entries[key] = value
                    added++
                }
            }
        }
        return added
    }

    @Suppress("UNCHECKED_CAST")
    private fun parse(line: String): Pair<String, VerifiedOutput> {
        val row = Json.parse(line) as Map<String, Any?>
        return row["key"] as String to VerifiedOutput(
            row["status"] as String,
            row["output"] as Map<String, Any?>,
            (row["contradictions"] as List<Any?>).map { it as String },
        )
    }

    /**
     * Save an answer. "partial" (the complete sentences written before Llama was stopped, already
     * checked) is kept until the full answer replaces it, and never replaces a full answer.
     */
    @Synchronized
    fun put(key: String, value: VerifiedOutput) {
        if (value.status !in SAVED) return
        val old = entries[key]
        if (value.status == "partial" && old != null && old.status != "partial") return
        if (old == value) return
        entries[key] = value
        val row = linkedMapOf("key" to key, "status" to value.status, "output" to value.output,
            "contradictions" to value.contradictions)
        file.appendText(Json.dumps(row) + "\n")
    }

    @Synchronized
    fun size(): Int = entries.size

    companion object {
        /** Bump when the prompt or the checks change, so old answers are not reused. */
        const val VERSION = 1
        private val SAVED = setOf("ok", "corrected", "partial")

        fun key(kbVersion: String, itemId: String, attributes: Map<String, Any?>, language: String): String {
            val attrs = attributes.filterValues { it is Boolean }.toSortedMap().entries.joinToString(",") { "${it.key}=${it.value}" }
            return "v$VERSION|$kbVersion|$itemId|$attrs|$language"
        }
    }
}
