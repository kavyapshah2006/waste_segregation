package wastesegregation.core

import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertEquals

/** tests/fixtures/voice_cases.json, the same cases as tests/test_voice.py. */
@Suppress("UNCHECKED_CAST")
class VoiceTest {
    private val kb = TestKb.kb
    private val data = TestKb.readJson("tests/fixtures/voice_cases.json") as Map<String, Any?>

    @TestFactory
    fun romanise() = (data["romanise"] as List<Map<String, Any?>>).map { case ->
        val text = case["text"] as String
        DynamicTest.dynamicTest("romanise $text") {
            assertEquals(case["expected"], listOf(Voice.romanise(text), Voice.key(text)))
        }
    }

    @TestFactory
    fun match() = (data["match"] as List<Map<String, Any?>>).map { case ->
        val transcript = case["transcript"] as String
        DynamicTest.dynamicTest("match '$transcript'") {
            val result = Voice.match(kb, transcript).map { it.toMap() }
            assertEquals(case["expected"], Json.parse(Json.dumps(result)))
        }
    }
}
