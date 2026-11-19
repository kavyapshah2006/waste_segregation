package wastesegregation.core

import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertEquals

/** tests/fixtures/agreement_cases.json, the same cases as test_agreement_rule in tests/test_fallback.py. */
@Suppress("UNCHECKED_CAST")
class AgreementTest {
    private val cases = TestKb.readJson("tests/fixtures/agreement_cases.json") as List<Map<String, Any?>>

    @TestFactory
    fun fixtures() = cases.map { case ->
        DynamicTest.dynamicTest(case["id"] as String) {
            val suggestions = (case["suggestions"] as List<String>).map { Suggestion(it, 0.1, 0.1) }
            assertEquals(case["expected"], FallbackRank.confirms(case["detector_item"] as String, suggestions))
        }
    }
}
