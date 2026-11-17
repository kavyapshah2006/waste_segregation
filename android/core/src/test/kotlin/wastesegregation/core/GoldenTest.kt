package wastesegregation.core

import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.TestFactory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The golden fixtures (tests/fixtures/golden) are the contract between the Python reference
 * and this port: verdict, exact prompt text and verified answer must all match. This mirrors
 * expected_for() in tests/fixtures/golden/regen.py.
 */
@Suppress("UNCHECKED_CAST")
class GoldenTest {
    private val kb = TestKb.kb
    private val files = TestKb.repoRoot.resolve("tests/fixtures/golden")
        .listFiles { f -> f.name.endsWith(".json") }!!
        .sortedBy { it.name }

    private fun expectedFor(input: Map<String, Any?>): Map<String, Any?> {
        val detection = input["detection"] as Map<String, Any?>
        val itemId = detection["item_id"] as String?
            ?: return linkedMapOf("verdict" to null, "prompt" to null, "verified" to null)
        val language = input["language"] as String? ?: "en"
        val attributes = detection["attributes"] as Map<String, Any?>? ?: emptyMap()
        val verdict = kb.resolve(itemId, attributes, language, input["city"] as String? ?: DEFAULT_CITY)
        val question = input["question"] as String?
        if (!Assistant.isOnTopic(question, kb.namesOf(itemId))) {
            return linkedMapOf(
                "verdict" to verdict.toMap(),
                "prompt" to null,
                "verified" to Assistant.declined(kb, verdict).toMap(),
            )
        }
        val prompt = Prompt.build(verdict, attributes, kb.languageName(language), question)
        val raw = input["raw_llm_output"] as String?
        return linkedMapOf(
            "verdict" to verdict.toMap(),
            "prompt" to prompt,
            "verified" to raw?.let { Verifier.verify(Prompt.fullAnswer(it), verdict).toMap() },
        )
    }

    @Test
    fun `there are enough fixtures`() {
        assertTrue(files.size >= 20, "only ${files.size} fixtures")
    }

    @TestFactory
    fun golden() = files.map { file ->
        DynamicTest.dynamicTest(file.nameWithoutExtension) {
            val case = Json.parse(file.readText(Charsets.UTF_8)) as Map<String, Any?>
            assertEquals(file.nameWithoutExtension, case["id"])
            val expected = case["expected"] as Map<String, Any?>
            val actual = expectedFor(case["input"] as Map<String, Any?>)
            // Compare the prompt on its own first: a diff of the full map is unreadable.
            assertEquals(expected["prompt"], actual["prompt"], "prompt differs")
            assertEquals(expected["verdict"], actual["verdict"], "verdict differs")
            assertEquals(expected["verified"], actual["verified"], "verified answer differs")
        }
    }
}
