package wastesegregation.core

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/** tests/fixtures/corrections_cases.json, the same cases as tests/test_corrections.py. */
@Suppress("UNCHECKED_CAST")
class CorrectionsTest {
    private val cases = TestKb.readJson("tests/fixtures/corrections_cases.json") as Map<String, Any?>

    private fun correction(row: Map<String, Any?>) = Correction(
        timeMs = (row["time_ms"] as JsonNumber).text.toLong(),
        source = row["source"] as String,
        predicted = row["predicted"] as String?,
        corrected = row["corrected"] as String,
        confidence = (row["confidence"] as JsonNumber?)?.text?.toDouble(),
        attributes = (row["attributes"] as Map<String, Any?>).mapValues { it.value as Boolean },
        image = row["image"] as String?,
    )

    private val rows = cases["corrections"] as List<Map<String, Any?>>
    private val lines = cases["lines"] as List<String>

    @Test
    fun linesMatchPython() {
        assertEquals(lines, rows.map { Corrections.toLine(correction(it)) })
    }

    @Test
    fun linesRoundTrip() {
        assertEquals(rows.map(::correction), lines.map(Corrections::parseLine))
    }

    @Test
    fun invalidRowsRejected() {
        for (row in cases["invalid"] as List<Map<String, Any?>>) {
            assertFailsWith<IllegalArgumentException>(row.toString()) { correction(row) }
        }
    }

    @Test
    fun logAppendsAndReads(@TempDir dir: File) {
        val log = CorrectionLog(File(dir, "sub/corrections.jsonl"))
        assertEquals(emptyList(), log.readAll())
        val all = rows.map(::correction)
        all.forEach(log::append)
        assertEquals(all, log.readAll())
        assertEquals(lines, File(dir, "sub/corrections.jsonl").readLines())
    }
}
