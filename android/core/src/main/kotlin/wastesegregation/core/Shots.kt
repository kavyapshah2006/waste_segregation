package wastesegregation.core

import java.util.regex.Pattern

/**
 * Worked examples for the Llama prompt: a port of pipeline/shots.py. Two finished exchanges
 * about fixed items (newspaper, AA battery), matched to the question's kind and language, come
 * before the real question. The golden fixtures pin the exact text on both sides.
 */
object Shots {
    val SHOT_ITEMS = listOf("newspaper", "aa_battery")
    val LANGUAGES = listOf("en", "hi")

    /** Canonical question per kind and language; null means the default question. */
    val QUESTIONS: Map<String, Map<String, String?>> = linkedMapOf(
        "explain" to mapOf("en" to null, "hi" to null),
        "which_bin" to mapOf("en" to "Which bin does this go in?", "hi" to "यह किस डिब्बे में जाएगा?"),
        "recyclable" to mapOf("en" to "Is this recyclable?", "hi" to "क्या यह रीसायकल हो सकता है?"),
        "prepare" to mapOf(
            "en" to "How should I prepare this before throwing it away?",
            "hi" to "इसे फेंकने से पहले कैसे तैयार करूँ?",
        ),
        "off_topic" to mapOf("en" to "What is the capital of France?", "hi" to "आज मौसम कैसा रहेगा?"),
    )

    private val HI_COLOUR_OBLIQUE = mapOf("green" to "हरे", "blue" to "नीले", "red" to "लाल", "black" to "काले")

    private val TIPS = mapOf(
        "wet" to mapOf(
            "en" to "Keep a small lidded bin in the kitchen for wet waste.",
            "hi" to "गीले कचरे के लिए रसोई में ढक्कन वाला छोटा डिब्बा रखें।",
        ),
        "dry" to mapOf(
            "en" to "Keep dry waste clean and dry so it is easy to sort.",
            "hi" to "सूखे कचरे को साफ़ और सूखा रखें ताकि उसे छाँटना आसान हो।",
        ),
        "sanitary" to mapOf(
            "en" to "Always wrap sanitary waste before putting it in the bin.",
            "hi" to "सैनिटरी कचरे को डिब्बे में डालने से पहले हमेशा लपेटें।",
        ),
        "special_care" to mapOf(
            "en" to "Store special care waste safely until the next collection drive.",
            "hi" to "विशेष देखभाल कचरे को अगले संग्रह अभियान तक सुरक्षित रखें।",
        ),
    )

    private const val FLAGS = Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE or Pattern.UNICODE_CHARACTER_CLASS
    private val KINDS = listOf(
        "recyclable" to Pattern.compile("recycl|रीसा[यइ]कल", FLAGS),
        "prepare" to Pattern.compile("""prepar|before\s+(?:throw|dispos|bin)|rinse|तैयार|पहले""", FLAGS),
        "which_bin" to Pattern.compile("""\b(?:dust)?bins?\b|डिब्ब|कूड़ेदान""", FLAGS),
    )

    /** explain (no question), recyclable, prepare, which_bin, or other. */
    fun questionKind(question: String?): String {
        if (question.isNullOrBlank()) return "explain"
        for ((kind, pattern) in KINDS) if (pattern.matcher(question).find()) return kind
        return "other"
    }

    fun shotKind(question: String?): String {
        val kind = questionKind(question)
        return if (kind in listOf("which_bin", "recyclable", "prepare")) kind else "explain"
    }

    private fun binPhrase(v: Verdict): String =
        if (v.language == "hi") "${HI_COLOUR_OBLIQUE[v.binColour] ?: v.binColourName} डिब्बे" else "${v.binColourName} bin"

    fun whyFor(kind: String, v: Verdict, kb: KnowledgeBase): String {
        val hi = v.language == "hi"
        if (kind == "off_topic") return kb.message("off_topic", v.language)
        val lead = when (kind) {
            "which_bin" -> if (hi) "यह ${binPhrase(v)} (${v.streamName}) में जाएगा।"
            else "It goes in the ${binPhrase(v)} (${v.streamName.lowercase()})."
            "recyclable" -> if (hi) {
                if (v.recyclable) "हाँ, इसे रीसायकल किया जा सकता है।" else "नहीं, इसे रीसायकल नहीं किया जा सकता।"
            } else {
                if (v.recyclable) "Yes, it can be recycled." else "No, it cannot be recycled."
            }
            "prepare" -> (if (hi) "पहला कदम: ${v.steps[0]}।" else "First: ${v.steps[0]}.").replace("।।", "।")
            else -> ""
        }
        val parts = listOf(lead) + v.notes + listOf(v.impact)
        return parts.filter { it.isNotEmpty() }.joinToString(" ").trim()
    }

    /** The whole answer object as JSON text, like json.dumps({"why", "tip"}, ensure_ascii=False). */
    fun answerFor(kind: String, v: Verdict, kb: KnowledgeBase): String {
        val tip = v.hazard ?: TIPS.getValue(v.stream).getValue(v.language)
        return Json.dumps(linkedMapOf("why" to whyFor(kind, v, kb), "tip" to tip))
    }

    /** (user message, full assistant answer) per example item; none outside English/Hindi. */
    fun shotTurns(kb: KnowledgeBase, question: String?, language: String): List<Pair<String, String>> {
        if (language !in LANGUAGES) return emptyList()
        val kind = shotKind(question)
        return SHOT_ITEMS.map { itemId ->
            val verdict = kb.resolve(itemId, emptyMap(), language)
            val user = Prompt.userMessage(verdict, emptyMap(), kb.languageName(language), QUESTIONS.getValue(kind)[language])
            user to answerFor(kind, verdict, kb)
        }
    }
}
