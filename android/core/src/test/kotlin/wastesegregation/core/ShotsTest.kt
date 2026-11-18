package wastesegregation.core

import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertEquals

/** tests/fixtures/question_kinds.json, the same cases as tests/test_shots.py. */
@Suppress("UNCHECKED_CAST")
class ShotsTest {
    private val cases = TestKb.readJson("tests/fixtures/question_kinds.json") as List<Map<String, Any?>>

    @TestFactory
    fun questionKinds() = cases.map { case ->
        val q = case["question"] as String?
        DynamicTest.dynamicTest("[$q]") {
            assertEquals(case["kind"], Shots.questionKind(q))
            assertEquals(case["shots"], Shots.shotKind(q))
        }
    }

    @Test
    fun exampleAnswersPassTheVerifier() {
        val kb = TestKb.kb
        for (language in listOf("en", "hi")) {
            for (question in listOf(null, "Is this recyclable?", "Which bin?", "Do I rinse it?")) {
                Shots.shotTurns(kb, question, language).zip(Shots.SHOT_ITEMS).forEach { (turn, itemId) ->
                    val verdict = kb.resolve(itemId, emptyMap(), language)
                    assertEquals("ok", Verifier.verify(turn.second, verdict).status, turn.second)
                }
            }
        }
        assertEquals(emptyList(), Shots.shotTurns(kb, null, "ta"))
    }
}
