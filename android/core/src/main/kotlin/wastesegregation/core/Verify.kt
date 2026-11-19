package wastesegregation.core

import java.util.regex.Pattern

/**
 * Checks Llama's answer against the knowledge base: a port of pipeline/verify.py (rules in
 * docs/FuturePlan.md section 5.5). The KB always wins on facts; only `why` and `tip` come from
 * Llama, and both are dropped if either contradicts the KB.
 *
 * Patterns use Unicode character classes (Regexes.UNICODE_CLASSES) so that \s, \S, \w and \b
 * follow Unicode like Python's `re` does for str patterns, on the JVM and on Android.
 */
data class VerifiedOutput(
    /** "ok", "corrected", "fallback" or "declined". */
    val status: String,
    val output: Map<String, Any?>,
    val contradictions: List<String>,
) {
    fun toMap(): Map<String, Any?> = linkedMapOf(
        "status" to status,
        "output" to output,
        "contradictions" to contradictions,
    )
}

private fun re(pattern: String, ignoreCase: Boolean = false): Pattern = Pattern.compile(
    pattern,
    Regexes.UNICODE_CLASSES or
        (if (ignoreCase) Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE else 0),
)

private fun Pattern.found(text: String) = matcher(text).find()

object Verifier {
    const val MAX_FREE_TEXT = 400

    // Bin mentions in free text, mapped to colour ids (Hindi direct and oblique forms).
    private val COLOUR_WORDS = mapOf(
        "green" to "green", "blue" to "blue", "red" to "red", "black" to "black",
        "yellow" to "yellow", "white" to "white", "grey" to "grey", "gray" to "grey",
        "हरा" to "green", "हरे" to "green", "नीला" to "blue", "नीले" to "blue",
        "लाल" to "red", "काला" to "black", "काले" to "black", "पीला" to "yellow",
        "पीले" to "yellow", "सफ़ेद" to "white", "सफेद" to "white",
    )
    private val EN_BIN = re("""\b(green|blue|red|black|yellow|white|grey|gray)\s+(?:dust)?bins?\b""", true)
    private val HI_BIN = re("""(हरा|हरे|नीला|नीले|लाल|काला|काले|पीला|पीले|सफ़ेद|सफेद)\s+(?:डिब्बे|डिब्बा|कूड़ेदान)""")

    private val SPECIAL_TOKENS = listOf("<|eot_id|>", "<|end_of_text|>")

    private fun tryObject(text: String): Map<String, Any?>? = try {
        @Suppress("UNCHECKED_CAST")
        Json.parse(text) as? Map<String, Any?>
    } catch (e: JsonParseException) {
        null
    }

    /**
     * The first JSON object in the text, ignoring prose and code fences. One repair: an object
     * that ends after a complete value but lacks its final "}" is closed. Text cut off inside a
     * string is never repaired.
     */
    fun extractJson(raw: String): Map<String, Any?>? {
        var text = raw
        for (token in SPECIAL_TOKENS) text = text.replace(token, "")
        var start = text.indexOf('{')
        while (start != -1) {
            var depth = 0
            var inString = false
            var escaped = false
            var closed = false
            for (i in start until text.length) {
                val ch = text[i]
                if (inString) {
                    if (escaped) escaped = false
                    else if (ch == '\\') escaped = true
                    else if (ch == '"') inString = false
                } else if (ch == '"') {
                    inString = true
                } else if (ch == '{') {
                    depth++
                } else if (ch == '}') {
                    depth--
                    if (depth == 0) {
                        tryObject(text.substring(start, i + 1))?.let { return it }
                        closed = true
                        break
                    }
                }
            }
            if (!closed && depth == 1 && !inString) {
                tryObject(text.substring(start).trimEnd() + "}")?.let { return it }
            }
            start = text.indexOf('{', start + 1)
        }
        return null
    }

    /** Colour ids of bins named in free text, e.g. 'blue bin' or 'नीले डिब्बे'. */
    fun binsMentioned(text: String): Set<String> {
        val found = mutableSetOf<String>()
        EN_BIN.matcher(text).let { m -> while (m.find()) found.add(COLOUR_WORDS.getValue(m.group(1).lowercase())) }
        HI_BIN.matcher(text).let { m -> while (m.find()) found.add(COLOUR_WORDS.getValue(m.group(1))) }
        return found
    }

    // ---- content checks, clause by clause -------------------------------------------------

    private val CLAUSE = re(
        """[.;:!?,।\n]+|\s+(?:and|but|or|so\s+that|because|और|लेकिन|या|ताकि|क्योंकि|जिससे)\s+""", true,
    )
    private val NEG_EN = re(
        """\b(?:not|never|no|don['’]t|doesn['’]t|shouldn['’]t|cannot|can['’]t|won['’]t|isn['’]t|""" +
            """aren['’]t|avoid)\b""",
        true,
    )
    private val NEG_HI = re("""नहीं|मत|(?<!\S)न(?!\S)""")
    private val STREAM_WORDS = mapOf(
        "wet" to (re("""\bwet\s+waste\b""", true) to re("""गील[ाे]\s+कचर[ाे]""")),
        "dry" to (re("""\bdry\s+waste\b""", true) to re("""सूख[ाे]\s+कचर[ाे]""")),
        "sanitary" to (re("""\bsanitary\s+waste\b""", true) to re("""सैनिटरी\s+कचर[ाे]""")),
        "special_care" to (
            re("""\bspecial[\s-]+care\s+waste\b""", true) to
                re("""विशेष\s+देखभाल\s+(?:वाले\s+)?कचर[ाे]""")
            ),
    )
    private const val RECYCL_HI = "रीसा[यइ]कल"
    private const val NOT_RECYCLABLE_HI = """$RECYCL_HI\s+(?:\S+\s+){0,3}?(?:नहीं|न)(?!\S)"""
    private val HARD = re(
        """मुश्किल|कठिन|शायद\s+ही|\bdifficult\b|\bhard\s+to\b|\brarely\b|\bseldom\b|\bhardly\b""", true,
    )
    private const val NOT_RECYCLABLE_EN =
        """\b(?:not|non-?|never|cannot|can['’]t|isn['’]t|aren['’]t)\s*(?:be\s+|being\s+)?recycl"""
    private const val RECYCLABLE_EN =
        """\b(?:is|are|it['’]s)\s+recyclable\b|\bcan\s+be\s+recycled\b|\b(?:is|are|gets?)\s+recycled\b"""
    private const val RECYCLABLE_HI = """$RECYCL_HI.*(?:हो\s+सकत|किया\s+जा\s+सकत|होत[ाीे]|हो\s+जात)"""
    private val RECYCLE_NEG = re("$NOT_RECYCLABLE_EN|$NOT_RECYCLABLE_HI", true)
    private val RECYCLE_POS = re("$RECYCLABLE_EN|$RECYCLABLE_HI", true)
    private val RECYCLE_TOPIC = re("recycl|$RECYCL_HI", true)

    private val NOT_COMPOSTABLE = re(
        """\b(?:not|non-?|never|cannot|can['’]t|isn['’]t|aren['’]t)\s*(?:be\s+)?compost""" +
            """|(?:खाद|कम्पोस्ट|कंपोस्ट)\s+(?:\S+\s+){0,3}?(?:नहीं|न)(?!\S)""",
        true,
    )
    private val COMPOSTABLE = re(
        """\b(?:is|are|it['’]s)\s+compostable\b|\bcan\s+be\s+composted\b""" +
            """|(?:खाद|कम्पोस्ट|कंपोस्ट)\s+(?:बन\s+सकत|बनाई\s+जा\s+सकत|बनत)""",
        true,
    )

    private val INSTEAD_OF_EN = re("""\binstead\s+of\b""", true) // negates what follows
    private val INSTEAD_OF_HI = re("""के\s+बजाय|की\s+जगह""") // negates what comes before

    private val UNSAFE_PLACE = re(
        """\b(?:drains?|sewers?|toilets?|nalas?|rivers?|lakes?|roadside|on\s+the\s+road|open\s+ground)\b""" +
            """|नाल[ाीियोंे]*|सीवर|शौचालय|सड़क|नदी|तालाब""",
        true,
    )
    private val PUT = re(
        """\b(?:put|pour|flush|throw|dump|tip|empty|drop|leave)\b""" +
            """|डालें|डाल\s+दें|बहाएँ|बहा\s+दें|फेंकें|फेंक\s+दें|छोड़ें|छोड़\s+दें""",
        true,
    )
    private val BURN = re("""(?:^|\byou\s+(?:can|should|may)\s+|\bplease\s+)burn\b|जलाएँ|जलाइए|जला\s+दें""", true)

    private val STOP = setOf(
        "the", "and", "or", "a", "an", "it", "its", "them", "they", "this", "that", "with",
        "from", "into", "onto", "before", "after", "your", "you", "out", "off", "for", "of",
        "to", "in", "on", "at", "be", "is", "are", "if", "can", "then", "so", "do", "not",
        "को", "से", "में", "और", "के", "की", "का", "है", "हैं", "यह", "इसे", "इस", "पहले",
        "दें", "करें", "कर", "या", "पर", "ही", "भी", "न", "नहीं", "मत",
    )
    private val WORD = re("""[\wऀ-ॿ]+""")

    internal fun clauses(text: String): List<String> =
        CLAUSE.split(text, -1).filter { it.isNotBlank() }

    private fun negated(clause: String) = NEG_EN.found(clause) || NEG_HI.found(clause)

    private fun namesVerdict(part: String, verdict: Verdict): Boolean {
        val (en, hi) = STREAM_WORDS.getValue(verdict.stream)
        return verdict.binColour in binsMentioned(part) || en.found(part) || hi.found(part)
    }

    /** True if a clause says not to use the verdict's own bin or stream. */
    fun negatesVerdict(text: String, verdict: Verdict): Boolean {
        for (clause in clauses(text)) {
            if (negated(clause) && namesVerdict(clause, verdict)) return true
            INSTEAD_OF_EN.matcher(clause).let { m ->
                if (m.find() && namesVerdict(clause.substring(m.end()), verdict)) return true
            }
            INSTEAD_OF_HI.matcher(clause).let { m ->
                if (m.find() && namesVerdict(clause.substring(0, m.start()), verdict)) return true
            }
        }
        return false
    }

    private fun claims(text: String, negative: Pattern, positive: Pattern): Set<Boolean> {
        val out = mutableSetOf<Boolean>()
        for (clause in clauses(text)) {
            if (negative.found(clause)) {
                out.add(false)
            } else if (!negated(clause) && !HARD.found(clause) && positive.found(clause)) {
                out.add(true)
            }
        }
        return out
    }

    /** Recyclability claims, including a bare leading "No," / "Yes," in a recycling text. */
    fun recyclableClaims(text: String): Set<Boolean> {
        val out = claims(text, RECYCLE_NEG, RECYCLE_POS).toMutableSet()
        val parts = clauses(text)
        if (parts.isNotEmpty() && RECYCLE_TOPIC.found(text)) {
            when (parts[0].trim().lowercase()) {
                "no", "नहीं" -> out.add(false)
                "yes", "हाँ", "हां" -> out.add(true)
            }
        }
        return out
    }

    fun compostableClaims(text: String): Set<Boolean> = claims(text, NOT_COMPOSTABLE, COMPOSTABLE)

    private fun contentWords(text: String): Set<String> {
        val out = mutableSetOf<String>()
        val m = WORD.matcher(text.lowercase())
        while (m.find()) {
            val w = m.group()
            if (w !in STOP && w.codePointCount(0, w.length) >= 3) out.add(w)
        }
        return out
    }

    private fun repeats(clause: String, kbClauses: List<Set<String>>, share: Double = 0.8): Boolean {
        val said = contentWords(clause)
        return kbClauses.any { words -> words.size >= 2 && (words intersect said).size >= share * words.size }
    }

    /** True if a negated clause matches a positive KB step ("Do not remove stickers"). */
    fun negatesStep(text: String, verdict: Verdict): Boolean {
        val stepWords = verdict.steps.flatMap { step ->
            clauses(step).filter { !negated(it) }.map { contentWords(it) }
        }
        return clauses(text).any { negated(it) && repeats(it, stepWords) }
    }

    /** True if a clause tells the user to put waste in a drain, toilet, road or river, or to
     * burn it; clauses that repeat the KB's own text are allowed. */
    fun unsafeInstruction(text: String, verdict: Verdict): Boolean {
        val kbClauses = (verdict.steps + verdict.notes + listOf(verdict.hazard ?: ""))
            .flatMap { clauses(it) }
            .map { contentWords(it) }
        for (clause in clauses(text)) {
            if (negated(clause) || repeats(clause, kbClauses)) continue
            if ((UNSAFE_PLACE.found(clause) && PUT.found(clause)) || BURN.found(clause.trim())) return true
        }
        return false
    }

    // ---- structured checks ------------------------------------------------------------------

    private val NORM_SPACE = re("""[\s\-]+""")

    private fun norm(value: String) = NORM_SPACE.matcher(value.trim().lowercase()).replaceAll("_")

    private fun streamKey(value: String) = norm(value).removeSuffix("_waste").removeSuffix("_कचरा")

    private fun matchesStream(value: Any?, verdict: Verdict) =
        streamKey(pyStr(value)) in setOf(verdict.stream, streamKey(verdict.streamName))

    private fun matchesBin(value: Any?, verdict: Verdict): Boolean {
        var v = norm(pyStr(value))
        for (suffix in listOf("_dustbin", "_bin", "_डिब्बा", "_डिब्बे")) v = v.removeSuffix(suffix)
        return v in setOf(verdict.binColour, norm(verdict.binColourName)) ||
            COLOUR_WORDS[v] == verdict.binColour
    }

    private fun cleanText(value: Any?): String? {
        if (value !is String) return null
        // Python's " ".join(value.split()): split on runs of Unicode whitespace.
        var text = value.splitWhere { it.isWhitespace() }.filter { it.isNotEmpty() }.joinToString(" ")
        if (text.isEmpty()) return null
        if (text.codePointCount(0, text.length) > MAX_FREE_TEXT) {
            val cut = text.substring(0, text.offsetByCodePoints(0, MAX_FREE_TEXT))
            text = cut.substringBeforeLast(" ") + "…"
        }
        return text
    }

    private inline fun String.splitWhere(isSeparator: (Char) -> Boolean): List<String> {
        val out = mutableListOf<String>()
        val sb = StringBuilder()
        for (ch in this) {
            if (isSeparator(ch)) {
                out.add(sb.toString())
                sb.setLength(0)
            } else {
                sb.append(ch)
            }
        }
        out.add(sb.toString())
        return out
    }

    /** The answer shown to the user: KB facts plus Llama's free text where it is allowed. */
    fun kbOutput(verdict: Verdict, why: String? = null, tip: String? = null): Map<String, Any?> = linkedMapOf(
        "item" to verdict.name,
        "stream" to verdict.stream,
        "bin" to verdict.binColour,
        "steps" to verdict.steps,
        "hazard" to verdict.hazard,
        "drop_off" to verdict.dropOffText,
        "why" to why,
        "tip" to tip,
        "language" to verdict.language,
    )

    /** Verify the complete answer text (use Prompt.fullAnswer on runner output first). */
    fun verify(raw: String, verdict: Verdict): VerifiedOutput {
        val data = extractJson(raw)
            ?: return VerifiedOutput("fallback", kbOutput(verdict), listOf("unparseable"))

        val contradictions = mutableListOf<String>()
        if ("stream" in data && !matchesStream(data["stream"], verdict)) contradictions.add("stream")
        if ("bin" in data && !matchesBin(data["bin"], verdict)) contradictions.add("bin")

        // The verdict's bin, plus bins the KB entry itself names, unless Llama got stream/bin wrong.
        val allowed = mutableSetOf(verdict.binColour)
        if (contradictions.isEmpty()) allowed += binsMentioned((verdict.steps + verdict.notes).joinToString(" "))

        val steps = data["steps"]
        if (steps is List<*>) {
            val stepText = steps.filterIsInstance<String>().joinToString(" ")
            if ((binsMentioned(stepText) - allowed).isNotEmpty()) contradictions.add("steps_bin")
        }

        val free = linkedMapOf<String, String?>()
        for (key in listOf("why", "tip")) {
            var value = cleanText(data[key])
            fun flag(name: String) {
                contradictions.add("${key}_$name")
                value = null
            }
            if (value != null && (binsMentioned(value!!) - allowed).isNotEmpty()) flag("bin")
            if (value != null && negatesVerdict(value!!, verdict)) flag("negates")
            if (value != null && (!verdict.recyclable) in recyclableClaims(value!!)) flag("recyclable")
            if (value != null && (!verdict.compostable) in compostableClaims(value!!)) flag("compostable")
            if (value != null && negatesStep(value!!, verdict)) flag("negates_step")
            if (value != null && unsafeInstruction(value!!, verdict)) flag("unsafe")
            free[key] = value
        }
        // One field wrong usually means the other repeats the mistake, so neither is shown.
        if (contradictions.any { it.startsWith("why_") || it.startsWith("tip_") }) {
            free["why"] = null
            free["tip"] = null
        }
        val status = if (contradictions.isEmpty()) "ok" else "corrected"
        return VerifiedOutput(status, kbOutput(verdict, free["why"], free["tip"]), contradictions.toList())
    }
}
