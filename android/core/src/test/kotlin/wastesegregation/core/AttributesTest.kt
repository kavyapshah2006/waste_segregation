package wastesegregation.core

import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/** tests/fixtures/attribute_cases.json, the same cases as tests/test_attributes.py. */
@Suppress("UNCHECKED_CAST")
class AttributesTest {
    private val fixture = TestKb.readJson("tests/fixtures/attribute_cases.json") as Map<String, Any?>
    private val checker = AttributeChecker(fixture["prompts"] as Map<String, Any?>)

    @TestFactory
    fun fixtures() = (fixture["cases"] as List<Map<String, Any?>>).map { case ->
        DynamicTest.dynamicTest(case["id"] as String) {
            val embedding = (case["embedding"] as List<Any?>).map { (it as JsonNumber).text.toFloat() }
            val result = checker.check(case["item_id"] as String, embedding.toFloatArray())
            val actual = Json.parse(Json.dumps(result.mapValues { it.value.toMap() }))
            assertEquals(Json.dumps(case["expected"]), Json.dumps(actual))
        }
    }

    @Test
    fun rejectsInvertedBand() {
        assertFailsWith<IllegalArgumentException> { AttributeChecker.decide(0.0, 0.1, -0.1) }
    }
}
