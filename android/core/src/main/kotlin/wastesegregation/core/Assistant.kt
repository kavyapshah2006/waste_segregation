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
            """collect\w*|pick\w*|burn\w*|broke\w*|break\w*|leak\w*|this|these|put\w*|go|goes|""" +
            """going|keep\w*|stor\w*|handl\w*|rid|old|empty|used|plastic\w*)\b""",
        FLAGS or Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE,
    )
    private val TOPIC_HI = Pattern.compile(
        """कचर|कूड़|डिब्ब|डब्ब|फेंक|रीसा[यइ]कल|खाद|कम्पोस्ट|कंपोस्ट|अलग|साफ़|साफ|धो|लपेट|सुरक्षित|""" +
            """ख़तर|खतर|ज़हर|जहर|दोबारा|बेच|कबाड़|टूट|डाल|रख|पुरान|प्लास्टिक|""" +
            """(?<!\S)(?:यह|ये|इसे|इस|इन|इसको)(?!\S)""",
        FLAGS,
    )

    /** KB name words too general to show a question is about waste (as pipeline/assistant.py). */
    private val VOCAB_STOP = (
        "item items small cold light soft water control pieces drink household band cell strip " +
            "pack with from other पानी किया"
        ).split(" ").toSet()
    private const val VOCAB_MIN = 4 // code points
    private const val FUZZY_MIN = 5 // words this long may differ by one letter

    private fun isWordChar(cp: Int): Boolean = when (Character.getType(cp).toByte()) {
        Character.UPPERCASE_LETTER, Character.LOWERCASE_LETTER, Character.TITLECASE_LETTER,
        Character.MODIFIER_LETTER, Character.OTHER_LETTER, Character.NON_SPACING_MARK,
        Character.ENCLOSING_MARK, Character.COMBINING_SPACING_MARK, Character.DECIMAL_DIGIT_NUMBER,
        Character.LETTER_NUMBER, Character.OTHER_NUMBER -> true
        else -> false
    }

    /** Lower-cased runs of letters, marks and digits (keeps Hindi vowel signs in the word). */
    fun words(text: String): List<String> {
        val out = mutableListOf<String>()
        val current = StringBuilder()
        text.lowercase().codePoints().forEach { cp ->
            if (isWordChar(cp)) {
                current.appendCodePoint(cp)
            } else if (current.isNotEmpty()) {
                out.add(current.toString())
                current.setLength(0)
            }
        }
        if (current.isNotEmpty()) out.add(current.toString())
        return out
    }

    private fun String.cpLength(): Int = codePointCount(0, length)

    /** Words of the KB's item names and aliases that count as waste words. */
    fun topicVocabulary(names: Iterable<String>): Set<String> =
        names.flatMap { words(it) }.filter { it.cpLength() >= VOCAB_MIN && it !in VOCAB_STOP }.toSet()

    /** True if a and b differ by at most one inserted, deleted or replaced letter (code points). */
    private fun oneEditApart(x: String, y: String): Boolean {
        var a = x.codePoints().toArray()
        var b = y.codePoints().toArray()
        if (kotlin.math.abs(a.size - b.size) > 1) return false
        if (a.size > b.size) a = b.also { b = a }
        var i = 0
        var j = 0
        var edits = 0
        while (i < a.size && j < b.size) {
            if (a[i] == b[j]) {
                i++
                j++
                continue
            }
            edits++
            if (edits > 1) return false
            if (a.size == b.size) i++
            j++
        }
        return edits + (b.size - j) + (a.size - i) <= 1
    }

    /**
     * False only for a question with no waste word, no item name and no KB name word.
     * [vocabulary]: [KnowledgeBase.topicVocabulary], the words of every item's names. The first
     * version knew only the word list and the item's names, and on the phone it declined
     * everyday questions such as "what do I do with an old plastic carton".
     */
    fun isOnTopic(question: String?, itemNames: List<String>, vocabulary: Set<String> = emptySet()): Boolean {
        if (question.isNullOrBlank()) return true
        if (TOPIC_EN.matcher(question).find() || TOPIC_HI.matcher(question).find()) return true
        val q = question.lowercase()
        if (itemNames.any { it.cpLength() >= 3 && it.lowercase() in q }) return true
        for (w in words(question)) {
            val n = w.cpLength()
            if (n < VOCAB_MIN) continue
            if (w in vocabulary) return true
            if (n >= FUZZY_MIN && vocabulary.any { it.cpLength() >= FUZZY_MIN && oneEditApart(w, it) }) return true
        }
        return false
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
        if (!isOnTopic(question, kb.namesOf(verdict.itemId), kb.topicVocabulary)) return Explanation(declined(kb, verdict))
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
