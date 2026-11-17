package wastesegregation.core

import java.math.BigDecimal
import java.math.RoundingMode
import kotlin.math.floor

/**
 * Impact summary of a user's scans: a port of pipeline/impact.py. All figures are estimates
 * from KB fields (typical item weights); the UI must label them so.
 */
data class Scan(val timeS: Double, val itemId: String, val attributes: Map<String, Any?> = emptyMap())

object Impact {
    const val DEDUPE_SECONDS = 30.0
    val STREAMS = listOf("wet", "dry", "sanitary", "special_care")
    private val SPECIAL_DROP_OFFS = setOf("e_waste_centre", "pharmacy_takeback", "hazardous_collection")
    private const val TOP_ITEMS = 5

    /** Counted scans in time order and the number of repeats dropped (same item < 30 s). */
    fun dedupe(scans: List<Scan>): Pair<List<Scan>, Int> {
        val counted = mutableListOf<Scan>()
        val last = mutableMapOf<String, Double>()
        for (scan in scans.sortedWith(compareBy({ it.timeS }, { it.itemId }))) {
            val prev = last[scan.itemId]
            if (prev != null && scan.timeS - prev < DEDUPE_SECONDS) continue
            counted.add(scan)
            last[scan.itemId] = scan.timeS
        }
        return counted to (scans.size - counted.size)
    }

    fun localDay(timeS: Double, tzOffsetMinutes: Int): Long =
        floor((timeS + tzOffsetMinutes * 60) / 86400.0).toLong()

    fun currentStreak(activeDays: Set<Long>, today: Long): Int {
        var day = if (today in activeDays) today else today - 1
        var streak = 0
        while (day in activeDays) {
            streak++
            day--
        }
        return streak
    }

    /** Python's round(x, 3): half-even on the exact binary value. */
    internal fun round3(x: Double): Double = BigDecimal(x).setScale(3, RoundingMode.HALF_EVEN).toDouble()

    fun summary(
        kb: KnowledgeBase,
        scans: List<Scan>,
        nowS: Double,
        tzOffsetMinutes: Int = 330,
        cityId: String = DEFAULT_CITY,
    ): Map<String, Any?> {
        val (counted, duplicates) = dedupe(scans)
        val byStream = STREAMS.associateWith { 0 }.toMutableMap()
        val items = mutableMapOf<String, Int>()
        var divertedG = 0.0
        var diverted = 0
        var neededCleaning = 0
        for (scan in counted) {
            val verdict = kb.resolve(scan.itemId, scan.attributes, "en", cityId)
            byStream[verdict.stream] = byStream.getValue(verdict.stream) + 1
            items[scan.itemId] = (items[scan.itemId] ?: 0) + 1
            if (scan.attributes["contaminated"] == true) neededCleaning++
            if (verdict.recyclable || verdict.compostable || verdict.dropOff in SPECIAL_DROP_OFFS) {
                diverted++
                divertedG += (kb.item(scan.itemId)["typical_weight_g"] as JsonNumber).text.toDouble()
            }
        }
        val days = counted.map { localDay(it.timeS, tzOffsetMinutes) }.toSet()
        val n = counted.size
        return linkedMapOf(
            "scans" to n,
            "duplicates_ignored" to duplicates,
            "by_stream" to LinkedHashMap(byStream),
            "top_items" to items.entries
                .sortedWith(compareBy({ -it.value }, { it.key }))
                .take(TOP_ITEMS)
                .map { listOf(it.key, it.value) },
            "diverted_items" to diverted,
            "diverted_kg_estimate" to round3(divertedG / 1000),
            "needed_cleaning_rate" to if (n > 0) round3(neededCleaning.toDouble() / n) else null,
            "active_days" to days.size,
            "current_streak_days" to currentStreak(days, localDay(nowS, tzOffsetMinutes)),
        )
    }
}

/**
 * Education content built only from KB facts: a port of pipeline/education.py. Rotation is
 * by day number, so Python and Kotlin show the same card on the same day.
 */
@Suppress("UNCHECKED_CAST")
object Education {
    fun didYouKnow(kb: KnowledgeBase, day: Long, language: String = "en"): Map<String, Any?> {
        val ids = kb.itemIds
        val itemId = ids[Math.floorMod(day, ids.size.toLong()).toInt()]
        val verdict = kb.resolve(itemId, emptyMap(), language)
        return linkedMapOf(
            "item_id" to itemId,
            "heading" to kb.message("did_you_know", language),
            "title" to verdict.name,
            "text" to verdict.impact,
        )
    }

    fun watchOutCards(kb: KnowledgeBase, language: String = "en"): List<Map<String, Any?>> {
        val cards = mutableListOf<Map<String, Any?>>()
        for (itemId in kb.itemIds) {
            for (rule in kb.item(itemId)["condition_rules"] as List<Map<String, Any?>>? ?: emptyList()) {
                if ("note" !in (rule["then"] as Map<String, Any?>)) continue
                val condition = LinkedHashMap(rule["if"] as Map<String, Any?>)
                val verdict = kb.resolve(itemId, condition, language)
                cards.add(
                    linkedMapOf(
                        "item_id" to itemId,
                        "condition" to condition,
                        "title" to verdict.name,
                        "stream" to verdict.stream,
                        "bin" to verdict.binColour,
                        "text" to verdict.notes.last(),
                    ),
                )
            }
        }
        return cards
    }

    fun quiz(kb: KnowledgeBase, day: Long, language: String = "en", cityId: String = DEFAULT_CITY): Map<String, Any?> {
        val ids = kb.itemIds
        val itemId = ids[Math.floorMod(day * 7 + 3, ids.size.toLong()).toInt()]
        val verdict = kb.resolve(itemId, emptyMap(), language, cityId)
        val colours = kb.city(cityId)["bin_colours"] as Map<String, Any?>
        val options = Impact.STREAMS.map { stream ->
            val colour = colours[stream] as String
            linkedMapOf(
                "stream" to stream,
                "bin" to colour,
                "label" to "${text(kb, "colours", colour, language)} (${text(kb, "streams", stream, language)})",
            )
        }
        return linkedMapOf(
            "item_id" to itemId,
            "question" to kb.message("quiz_which_bin", language).replace("{item}", verdict.name),
            "options" to options,
            "answer" to verdict.stream,
            "explanation" to verdict.impact,
        )
    }

    private fun text(kb: KnowledgeBase, section: String, key: String, language: String): String {
        val byLang = (kb.strings[section] as Map<String, Any?>)[key] as Map<String, Any?>
        return (byLang[language] ?: byLang.getValue("en")) as String
    }
}
