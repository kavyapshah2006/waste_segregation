package wastesegregation.core

/**
 * A small JSON reader and writer that behave like Python's `json` module, which the
 * reference pipeline uses. Exact parity matters here: the prompt text is compared byte for
 * byte against the golden fixtures, and the verifier must accept and reject the same model
 * output as Python does.
 *
 * Values are plain Kotlin types: Map<String, Any?> (insertion-ordered), List<Any?>, String,
 * Boolean, JsonNumber and null.
 */
data class JsonNumber(val text: String) {
    override fun toString(): String = text
}

class JsonParseException(message: String) : RuntimeException(message)

object Json {
    /** Like Python's json.loads (strict mode): raises JsonParseException on any error. */
    fun parse(text: String): Any? {
        val p = Parser(text)
        p.skipWhitespace()
        val value = p.value()
        p.skipWhitespace()
        if (p.pos != text.length) throw JsonParseException("extra data at ${p.pos}")
        return value
    }

    /** Like json.dumps(value, ensure_ascii=False, separators=(", ", ": ")). */
    fun dumps(value: Any?): String = StringBuilder().also { write(it, value) }.toString()

    private fun write(out: StringBuilder, value: Any?) {
        when (value) {
            null -> out.append("null")
            is Boolean -> out.append(if (value) "true" else "false")
            is String -> writeString(out, value)
            is JsonNumber -> out.append(value.text)
            is Int, is Long -> out.append(value.toString())
            is Map<*, *> -> {
                out.append('{')
                var first = true
                for ((k, v) in value) {
                    if (!first) out.append(", ")
                    first = false
                    writeString(out, k as String)
                    out.append(": ")
                    write(out, v)
                }
                out.append('}')
            }
            is List<*> -> {
                out.append('[')
                value.forEachIndexed { i, v ->
                    if (i > 0) out.append(", ")
                    write(out, v)
                }
                out.append(']')
            }
            else -> throw IllegalArgumentException("cannot write ${value::class}")
        }
    }

    private fun writeString(out: StringBuilder, s: String) {
        out.append('"')
        for (ch in s) {
            when (ch) {
                '"' -> out.append("\\\"")
                '\\' -> out.append("\\\\")
                '\n' -> out.append("\\n")
                '\r' -> out.append("\\r")
                '\t' -> out.append("\\t")
                '\b' -> out.append("\\b")
                '\u000c' -> out.append("\\f")
                else -> if (ch < ' ') out.append("\\u%04x".format(ch.code)) else out.append(ch)
            }
        }
        out.append('"')
    }

    private class Parser(val s: String) {
        var pos = 0

        fun skipWhitespace() {
            while (pos < s.length && s[pos] in " \t\n\r") pos++
        }

        fun fail(what: String): Nothing = throw JsonParseException("$what at $pos")

        fun value(): Any? {
            if (pos >= s.length) fail("unexpected end")
            return when (val ch = s[pos]) {
                '{' -> obj()
                '[' -> array()
                '"' -> string()
                't' -> literal("true", true)
                'f' -> literal("false", false)
                'n' -> literal("null", null)
                'N' -> literal("NaN", JsonNumber("NaN"))
                'I' -> literal("Infinity", JsonNumber("Infinity"))
                else -> if (ch == '-' || ch.isAsciiDigit()) number() else fail("unexpected '$ch'")
            }
        }

        private fun Char.isAsciiDigit() = this in '0'..'9'

        private fun literal(word: String, result: Any?): Any? {
            if (!s.startsWith(word, pos)) fail("bad literal")
            pos += word.length
            return result
        }

        private fun obj(): Map<String, Any?> {
            pos++ // {
            val out = LinkedHashMap<String, Any?>()
            skipWhitespace()
            if (pos < s.length && s[pos] == '}') {
                pos++
                return out
            }
            while (true) {
                skipWhitespace()
                if (pos >= s.length || s[pos] != '"') fail("expected key")
                val key = string()
                skipWhitespace()
                if (pos >= s.length || s[pos] != ':') fail("expected ':'")
                pos++
                skipWhitespace()
                out[key] = value() // a repeated key keeps its first position, last value wins
                skipWhitespace()
                if (pos >= s.length) fail("unterminated object")
                when (s[pos]) {
                    ',' -> pos++
                    '}' -> {
                        pos++
                        return out
                    }
                    else -> fail("expected ',' or '}'")
                }
            }
        }

        private fun array(): List<Any?> {
            pos++ // [
            val out = ArrayList<Any?>()
            skipWhitespace()
            if (pos < s.length && s[pos] == ']') {
                pos++
                return out
            }
            while (true) {
                skipWhitespace()
                out.add(value())
                skipWhitespace()
                if (pos >= s.length) fail("unterminated array")
                when (s[pos]) {
                    ',' -> pos++
                    ']' -> {
                        pos++
                        return out
                    }
                    else -> fail("expected ',' or ']'")
                }
            }
        }

        private fun string(): String {
            pos++ // opening quote
            val out = StringBuilder()
            while (true) {
                if (pos >= s.length) fail("unterminated string")
                val ch = s[pos++]
                when {
                    ch == '"' -> return out.toString()
                    ch == '\\' -> {
                        if (pos >= s.length) fail("bad escape")
                        when (val e = s[pos++]) {
                            '"' -> out.append('"')
                            '\\' -> out.append('\\')
                            '/' -> out.append('/')
                            'b' -> out.append('\b')
                            'f' -> out.append('\u000c')
                            'n' -> out.append('\n')
                            'r' -> out.append('\r')
                            't' -> out.append('\t')
                            'u' -> out.append(hex4())
                            else -> fail("bad escape '\\$e'")
                        }
                    }
                    ch < ' ' -> fail("control character in string") // Python strict mode
                    else -> out.append(ch)
                }
            }
        }

        private fun hex4(): Char {
            if (pos + 4 > s.length) fail("bad \\u escape")
            val hex = s.substring(pos, pos + 4)
            if (!hex.all { it.isAsciiDigit() || it.lowercaseChar() in 'a'..'f' }) fail("bad \\u escape")
            pos += 4
            return hex.toInt(16).toChar()
        }

        private fun number(): JsonNumber {
            val start = pos
            if (s[pos] == '-') {
                pos++
                if (s.startsWith("Infinity", pos)) {
                    pos += "Infinity".length
                    return JsonNumber("-Infinity")
                }
            }
            if (pos >= s.length || !s[pos].isAsciiDigit()) fail("bad number")
            if (s[pos] == '0') pos++ else while (pos < s.length && s[pos].isAsciiDigit()) pos++
            if (pos + 1 < s.length && s[pos] == '.' && s[pos + 1].isAsciiDigit()) {
                pos++
                while (pos < s.length && s[pos].isAsciiDigit()) pos++
            }
            if (pos < s.length && (s[pos] == 'e' || s[pos] == 'E')) {
                var p = pos + 1
                if (p < s.length && (s[p] == '+' || s[p] == '-')) p++
                if (p < s.length && s[p].isAsciiDigit()) {
                    while (p < s.length && s[p].isAsciiDigit()) p++
                    pos = p
                }
            }
            return JsonNumber(s.substring(start, pos))
        }
    }
}
