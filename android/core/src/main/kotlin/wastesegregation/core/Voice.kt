package wastesegregation.core

import java.math.BigDecimal
import java.math.RoundingMode
import java.text.Normalizer

/**
 * Match a speech transcript to KB items: a port of pipeline/voice.py. Recognisers write Hindi
 * names in Latin or another script, or with small slips, so items are matched on a rough
 * pronunciation key with edit-distance similarity, not by exact text.
 */
data class VoiceMatch(val itemId: String, val score: Double, val confirm: Boolean) {
    fun toMap(): Map<String, Any?> = linkedMapOf("item_id" to itemId, "score" to score, "confirm" to confirm)
}

object Voice {
    const val CONFIRM_BELOW = 0.85
    const val MIN_SCORE = 0.5
    const val MIN_WINDOW_NAME = 5
    const val CONFIDENT_MARGIN = 0.1

    private val CONSONANTS = mapOf(
        'क' to "k", 'ख' to "kh", 'ग' to "g", 'घ' to "gh", 'ङ' to "n", 'च' to "ch", 'छ' to "chh", 'ज' to "j",
        'झ' to "jh", 'ञ' to "n", 'ट' to "t", 'ठ' to "th", 'ड' to "d", 'ढ' to "dh", 'ण' to "n", 'त' to "t",
        'थ' to "th", 'द' to "d", 'ध' to "dh", 'न' to "n", 'प' to "p", 'फ' to "ph", 'ब' to "b", 'भ' to "bh",
        'म' to "m", 'य' to "y", 'र' to "r", 'ल' to "l", 'व' to "v", 'श' to "sh", 'ष' to "sh", 'स' to "s",
        'ह' to "h", 'ळ' to "l",
    )
    private val VOWELS = mapOf(
        'अ' to "a", 'आ' to "aa", 'इ' to "i", 'ई' to "ii", 'उ' to "u", 'ऊ' to "uu", 'ऋ' to "ri", 'ए' to "e",
        'ऐ' to "ai", 'ओ' to "o", 'औ' to "au", 'ऑ' to "o",
    )
    private val MATRAS = mapOf(
        'ा' to "aa", 'ि' to "i", 'ी' to "ii", 'ु' to "u", 'ू' to "uu", 'ृ' to "ri", 'े' to "e", 'ै' to "ai",
        'ो' to "o", 'ौ' to "au", 'ॉ' to "o",
    )
    private val NUKTA_MAP = mapOf('ड' to "r", 'ढ' to "rh", 'फ' to "f", 'ज' to "j", 'क' to "k", 'ख' to "kh", 'ग' to "g")
    private const val VIRAMA = '्'
    private const val NUKTA = '़'
    private val NASAL = mapOf('ं' to "n", 'ँ' to "n", 'ः' to "h")
    private val URDU = mapOf(
        "ا" to "a", "آ" to "a", "ب" to "b", "پ" to "p", "ت" to "t", "ٹ" to "t", "ث" to "s", "ج" to "j", "چ" to "ch",
        "ح" to "h", "خ" to "kh", "د" to "d", "ڈ" to "d", "ذ" to "z", "ر" to "r", "ڑ" to "r", "ز" to "z", "ژ" to "zh",
        "س" to "s", "ش" to "sh", "ص" to "s", "ض" to "z", "ط" to "t", "ظ" to "z", "غ" to "gh", "ف" to "f", "ق" to "k",
        "ک" to "k", "ك" to "k", "گ" to "g", "ل" to "l", "م" to "m", "ن" to "n", "ں" to "n", "و" to "v", "ہ" to "h",
        "ھ" to "h", "ة" to "h", "ی" to "i", "ي" to "i", "ے" to "e", "ئ" to "i",
    )
    private val SIMPLIFY = listOf(
        "aa" to "a", "ii" to "i", "ee" to "i", "uu" to "u", "oo" to "u", "ai" to "e", "au" to "o",
        "ph" to "f", "w" to "v", "z" to "j", "q" to "k", "c" to "k", "y" to "i",
    )

    private fun isDevanagari(c: Char?) = c != null && c in 'ऀ'..'ॿ'

    /** Other Indic scripts back to Devanagari by position; Urdu letters to Latin. */
    private fun toDevanagari(text: String): String {
        val sb = StringBuilder()
        text.codePoints().forEach { cp ->
            val mapped = if (cp in 0x0980..0x0D7F) 0x0900 + (cp - 0x0980) % 0x80 else cp
            val s = String(Character.toChars(mapped))
            sb.append(URDU[s] ?: s)
        }
        return sb.toString()
    }

    fun romanise(input: String): String {
        val text = Normalizer.normalize(
            toDevanagari(Normalizer.normalize(input, Normalizer.Form.NFC)),
            Normalizer.Form.NFD,
        )
        val out = mutableListOf<String>()
        var i = 0
        while (i < text.length) {
            val ch = text[i]
            val consonant = CONSONANTS[ch]
            if (consonant != null) {
                var sound = consonant
                var nxt = text.getOrNull(i + 1)
                if (nxt == NUKTA) {
                    sound = NUKTA_MAP[ch] ?: sound
                    i++
                    nxt = text.getOrNull(i + 1)
                }
                val after = text.getOrNull(i + 2)
                val prevVowel = out.isNotEmpty() && out.last().last() in "aeiou"
                out.add(sound)
                when {
                    nxt != null && nxt in MATRAS -> {
                        out.add(MATRAS.getValue(nxt))
                        i++
                    }
                    nxt == VIRAMA -> i++
                    !isDevanagari(nxt) || nxt in NASAL -> Unit // word end: final "a" is silent
                    prevVowel && nxt in CONSONANTS && after != null && after in MATRAS -> Unit // schwa deletion
                    else -> out.add("a")
                }
            } else if (ch in VOWELS) {
                out.add(VOWELS.getValue(ch))
            } else if (ch in NASAL) {
                out.add(NASAL.getValue(ch))
            } else if (ch in MATRAS) {
                out.add(MATRAS.getValue(ch))
            } else if (ch != NUKTA && ch != VIRAMA) {
                out.add(ch.toString())
            }
            i++
        }
        return out.joinToString("")
    }

    private val NON_LETTERS = Regex("[^a-z]")
    private val DOUBLES = Regex("(.)\\1+")

    fun key(text: String): String {
        var s = NON_LETTERS.replace(romanise(text).lowercase(), "")
        s = s.replace("chh", "ch").replace("kh", "k").replace("gh", "g").replace("th", "t")
            .replace("dh", "d").replace("bh", "b").replace("sh", "s")
        for ((a, b) in SIMPLIFY) s = s.replace(a, b)
        return DOUBLES.replace(s, "$1")
    }

    fun similarity(a: String, b: String): Double {
        if (a.isEmpty() && b.isEmpty()) return 1.0
        var prev = IntArray(b.length + 1) { it }
        for (i in 1..a.length) {
            val cur = IntArray(b.length + 1)
            cur[0] = i
            for (j in 1..b.length) {
                cur[j] = minOf(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + if (a[i - 1] != b[j - 1]) 1 else 0)
            }
            prev = cur
        }
        return 1 - prev[b.length].toDouble() / maxOf(a.length, b.length)
    }

    fun windowSimilarity(heard: String, name: String): Double {
        val full = similarity(heard, name)
        if (name.length < MIN_WINDOW_NAME || heard.length <= name.length) return full
        var best = full
        for (width in listOf(name.length - 1, name.length, name.length + 1)) {
            for (start in 0..(heard.length - width)) {
                best = maxOf(best, similarity(heard.substring(start, start + width), name))
            }
        }
        return best
    }

    fun match(kb: KnowledgeBase, transcript: String, k: Int = 3): List<VoiceMatch> {
        val heard = key(transcript)
        if (heard.isEmpty()) return emptyList()
        val best = kb.itemIds.associateWith { id -> kb.namesOf(id).maxOf { windowSimilarity(heard, key(it)) } }
        val ranked = best.entries.sortedWith(compareBy({ -it.value }, { it.key }))
        val clear = ranked.size < 2 || ranked[0].value - ranked[1].value >= CONFIDENT_MARGIN
        return ranked.take(k).withIndex()
            .filter { it.value.value >= MIN_SCORE }
            .map { (rank, e) ->
                VoiceMatch(e.key, round3(e.value), !(rank == 0 && clear && e.value >= CONFIRM_BELOW))
            }
    }

    private fun round3(x: Double) = BigDecimal(x).setScale(3, RoundingMode.HALF_EVEN).toDouble()
}
