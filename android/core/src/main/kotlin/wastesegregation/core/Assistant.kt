package wastesegregation.core

import java.util.regex.Pattern

/**
 * Asks Llama to explain a verdict, with one shorter retry, then falls back to KB text: a port
 * of pipeline/assistant.py, including the off-topic guard that declines a question without
 * calling Llama.
 */
data class Explanation(
    val verified: VerifiedOutput,
    val prompts: List<String> = emptyList(),
    val rawOutputs: List<String> = emptyList(),
) {
    val attempts: Int get() = prompts.size
}

object Assistant {
    private val FLAGS = Regexes.UNICODE_CLASSES

    private val TOPIC_EN = Pattern.compile(
        """\b(?:bins?|dustbins?|waste|trash|garbage|rubbish|junk|recycl\w*|compost\w*|dispos\w*|""" +
            """throw\w*|thrown|dump\w*|sort\w*|segregat\w*|separat\w*|bags?|clean\w*|rinse|wash\w*|""" +
            """wrap\w*|safe\w*|danger\w*|harm\w*|toxic|hazard\w*|reus\w*|sell|scrap|kabadi\w*|""" +
            """collect\w*|pick\w*|burn\w*|broke\w*|break\w*|leak\w*|this|these)\b""",
        FLAGS or Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE,
    )
    private val TOPIC_HI = Pattern.compile(
        """कचर|कूड़|डिब्ब|फेंक|रीसा[यइ]कल|खाद|कम्पोस्ट|कंपोस्ट|अलग|साफ़|साफ|धो|लपेट|सुरक्षित|""" +
            """ख़तर|खतर|ज़हर|जहर|दोबारा|बेच|कबाड़|टूट|(?<!\S)(?:यह|ये|इसे|इस|इन|इसको)(?!\S)""",
        FLAGS,
    )

    /** False only for a question that uses no waste word and does not name the item. */
    fun isOnTopic(question: String?, itemNames: List<String>): Boolean {
        if (question.isNullOrBlank()) return true
        if (TOPIC_EN.matcher(question).find() || TOPIC_HI.matcher(question).find()) return true
        val q = question.lowercase()
        return itemNames.any { it.codePointCount(0, it.length) >= 3 && it.lowercase() in q }
    }

    /** The answer for an off-topic question: KB facts plus a fixed note, no Llama text. */
    fun declined(kb: KnowledgeBase, verdict: Verdict): VerifiedOutput = VerifiedOutput(
        "declined",
        Verifier.kbOutput(verdict, why = kb.message("off_topic", verdict.language)),
        emptyList(),
    )

    /** Extra sampled tries when the greedy answer fails the checks (as pipeline/assistant.py). */
    const val SAMPLED_RETRIES = 2

    /**
     * [generate] takes the full prompt and returns the text the model generated after it
     * (greedy). Unparseable -> one short-prompt retry; still not "ok" and [sample] given -> up
     * to [SAMPLED_RETRIES] sampled tries, the first "ok" wins.
     */
    fun explain(
        kb: KnowledgeBase,
        verdict: Verdict,
        attributes: Map<String, Any?>,
        generate: (String) -> String,
        question: String? = null,
        includeBos: Boolean = true,
        sample: ((String) -> String)? = null,
    ): Explanation {
        if (!isOnTopic(question, kb.namesOf(verdict.itemId))) return Explanation(declined(kb, verdict))
        val languageName = kb.languageName(verdict.language)
        val prompts = mutableListOf<String>()
        val raws = mutableListOf<String>()

        fun attempt(gen: (String) -> String, short: Boolean): VerifiedOutput {
            val prompt = Prompt.build(verdict, attributes, languageName, question, short, includeBos, kb)
            val raw = gen(prompt)
            prompts.add(prompt)
            raws.add(raw)
            return Verifier.verify(Prompt.fullAnswer(raw), verdict)
        }

        var best = attempt(generate, false)
        if (best.status == "fallback") {
            val second = attempt(generate, true)
            if (second.status != "fallback") best = second
        }
        if (best.status != "ok" && sample != null) {
            repeat(SAMPLED_RETRIES) {
                if (best.status == "ok") return@repeat
                val tried = attempt(sample, false)
                if (tried.status == "ok" || (best.status == "fallback" && tried.status == "corrected")) best = tried
            }
        }
        return Explanation(best, prompts, raws)
    }
}
