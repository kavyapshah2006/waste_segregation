package wastesegregation.core

import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.TestFactory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/** The same per-function cases as tests/test_check_cases.py (tests/fixtures/check_cases.json). */
@Suppress("UNCHECKED_CAST")
class CheckCasesTest {
    private val kb = TestKb.kb
    private val cases = TestKb.readJson("tests/fixtures/check_cases.json") as Map<String, Any?>

    private fun section(name: String, check: (Map<String, Any?>) -> Unit) =
        (cases.getValue(name) as List<Map<String, Any?>>).mapIndexed { i, case ->
            val label = (case["text"] ?: case["question"]).toString().take(50)
            DynamicTest.dynamicTest("$name[$i] $label") { check(case) }
        }

    private fun verdict(case: Map<String, Any?>) = kb.resolve(
        case["item"] as String,
        case["attributes"] as Map<String, Any?>? ?: emptyMap(),
        case["language"] as String? ?: "en",
    )

    private fun sortedBools(value: Set<Boolean>) = value.sorted()

    @TestFactory
    fun extractJson() = section("extract_json") {
        assertEquals(it["expected"], Verifier.extractJson(it["text"] as String))
    }

    @TestFactory
    fun binsMentioned() = section("bins_mentioned") {
        assertEquals(it["expected"], Verifier.binsMentioned(it["text"] as String).sorted())
    }

    @TestFactory
    fun negatesVerdict() = section("negates_verdict") {
        assertEquals(it["expected"], Verifier.negatesVerdict(it["text"] as String, verdict(it)))
    }

    @TestFactory
    fun recyclableClaims() = section("recyclable_claims") {
        assertEquals(it["expected"], sortedBools(Verifier.recyclableClaims(it["text"] as String)))
    }

    @TestFactory
    fun compostableClaims() = section("compostable_claims") {
        assertEquals(it["expected"], sortedBools(Verifier.compostableClaims(it["text"] as String)))
    }

    @TestFactory
    fun negatesStep() = section("negates_step") {
        assertEquals(it["expected"], Verifier.negatesStep(it["text"] as String, verdict(it)))
    }

    @TestFactory
    fun unsafeInstruction() = section("unsafe_instruction") {
        assertEquals(it["expected"], Verifier.unsafeInstruction(it["text"] as String, verdict(it)))
    }

    @TestFactory
    fun isOnTopic() = section("is_on_topic") {
        assertEquals(it["expected"], Assistant.isOnTopic(it["question"] as String?, kb.namesOf(it["item"] as String)))
    }

    @TestFactory
    fun search() = section("search") {
        assertEquals(it["expected"], kb.search(it["text"] as String))
    }

    @Test
    fun `json writer matches python dumps`() {
        val value = linkedMapOf("a" to "x\"\\\n\u0001नी", "b" to listOf(true, null), "c" to emptyMap<String, Any?>())
        assertEquals("{\"a\": \"x\\\"\\\\\\n\\u0001नी\", \"b\": [true, null], \"c\": {}}", Json.dumps(value))
    }

    @Test
    fun `json reader is strict like python`() {
        assertFailsWith<JsonParseException> { Json.parse("{'a': 1}") }
        assertFailsWith<JsonParseException> { Json.parse("[1] x") }
        assertEquals(listOf(JsonNumber("-1.5e3")), Json.parse(" [-1.5e3] "))
    }

    @Test
    fun `explain retries once then falls back`() {
        val verdict = kb.resolve("aa_battery")
        val outputs = ArrayDeque(listOf("cut off mid-sent", "Batteries leak metals.\", \"tip\": \"Tape the ends.\"}"))
        val result = Assistant.explain(kb, verdict, emptyMap(), { outputs.removeFirst() })
        assertEquals(2, result.attempts)
        assertEquals("ok", result.verified.status)
        val failed = Assistant.explain(kb, verdict, emptyMap(), { "still not json" })
        assertEquals("fallback", failed.verified.status)
        assertEquals(verdict.steps, failed.verified.output["steps"])
    }

    @Test
    fun `explain samples after a contradiction, as in Python`() {
        val verdict = kb.resolve("aa_battery")
        val wrong = "Use the green bin.\", \"tip\": \"x\"}"
        val good = "Batteries leak metals.\", \"tip\": \"Tape them.\"}"
        val sampled = ArrayDeque(listOf(wrong, good))
        val result = Assistant.explain(kb, verdict, emptyMap(), { wrong }, sample = { sampled.removeFirst() })
        assertEquals(3, result.attempts)
        assertEquals("ok", result.verified.status)
        val kept = Assistant.explain(kb, verdict, emptyMap(), { "not json" }, sample = { wrong })
        assertEquals(4, kept.attempts)
        assertEquals("corrected", kept.verified.status)
        val once = Assistant.explain(kb, verdict, emptyMap(), { good }, sample = { error("no sampling after ok") })
        assertEquals(1, once.attempts)
    }

    @Test
    fun `off topic question never calls llama`() {
        val verdict = kb.resolve("pet_bottle", language = "hi")
        val result = Assistant.explain(kb, verdict, emptyMap(), { error("Llama must not be called") }, "आज मौसम कैसा रहेगा?")
        assertEquals(0, result.attempts)
        assertEquals("declined", result.verified.status)
        assertEquals(kb.message("off_topic", "hi"), result.verified.output["why"])
    }
}
