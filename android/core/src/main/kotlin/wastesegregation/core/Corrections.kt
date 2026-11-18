package wastesegregation.core

import java.io.File

/**
 * One "This is wrong" correction (FuturePlan WP7 step 6). A port of pipeline/corrections.py:
 * [toLine] must write exactly what Python's to_line writes (tests/fixtures/corrections_cases.json).
 * The log stays on the phone; the user can export it to help relabel and retrain.
 */
data class Correction(
    val timeMs: Long,
    val source: String,
    val predicted: String?,
    val corrected: String,
    val confidence: Double? = null,
    val attributes: Map<String, Boolean> = emptyMap(),
    val image: String? = null,
) {
    init {
        require(source in Corrections.SOURCES) { "unknown source $source" }
        require(corrected.isNotEmpty()) { "corrected item is empty" }
        require(confidence == null || confidence in 0.0..1.0) { "confidence out of range: $confidence" }
        val unknown = attributes.keys - Corrections.ATTRIBUTES.toSet()
        require(unknown.isEmpty()) { "unknown attributes ${unknown.sorted()}" }
    }
}

object Corrections {
    val SOURCES = listOf("detector", "fallback", "voice", "photo")
    val ATTRIBUTES = listOf("broken", "contaminated")

    fun toLine(c: Correction): String {
        val attributes = linkedMapOf<String, Any?>()
        for (k in ATTRIBUTES) c.attributes[k]?.let { attributes[k] = it }
        val row = linkedMapOf<String, Any?>(
            "time_ms" to c.timeMs,
            "source" to c.source,
            "predicted" to c.predicted,
            "corrected" to c.corrected,
            "confidence" to c.confidence,
            "attributes" to attributes,
            "image" to c.image,
        )
        return Json.dumps(row)
    }

    @Suppress("UNCHECKED_CAST")
    fun parseLine(line: String): Correction {
        val row = Json.parse(line) as Map<String, Any?>
        return Correction(
            timeMs = (row["time_ms"] as JsonNumber).text.toLong(),
            source = row["source"] as String,
            predicted = row["predicted"] as String?,
            corrected = row["corrected"] as String,
            confidence = (row["confidence"] as JsonNumber?)?.text?.toDouble(),
            attributes = (row["attributes"] as Map<String, Any?>? ?: emptyMap()).mapValues { it.value as Boolean },
            image = row["image"] as String?,
        )
    }
}

/** Appends corrections to a JSON-lines file in the app's private storage. */
class CorrectionLog(private val file: File) {
    fun append(c: Correction) {
        file.parentFile?.mkdirs()
        file.appendText(Corrections.toLine(c) + "\n", Charsets.UTF_8)
    }

    fun readAll(): List<Correction> =
        if (!file.exists()) emptyList()
        else file.readLines(Charsets.UTF_8).filter { it.isNotBlank() }.map(Corrections::parseLine)
}
