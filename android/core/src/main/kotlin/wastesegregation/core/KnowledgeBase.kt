package wastesegregation.core

/**
 * Knowledge base lookup and decision resolution: a port of pipeline/kb.py.
 *
 * The data comes from the compiled kb.sqlite (kb/build_kb.py). Reading the database is
 * platform code (JDBC in the tests, android.database on the phone), so this class takes the
 * already-loaded rows.
 */
const val DEFAULT_CITY = "default_in"
private const val FALLBACK_LANG = "en"

data class AliasRow(val alias: String, val lang: String, val itemId: String)

data class KbData(
    /** item id -> the item's JSON (items.json column). */
    val items: Map<String, Map<String, Any?>>,
    /** city id -> the city's JSON (cities.json column). */
    val cities: Map<String, Map<String, Any?>>,
    val aliases: List<AliasRow>,
    /** detector class -> item id. */
    val detectorClasses: Map<String, String>,
    /** meta.kb_version */
    val version: String,
    /** meta.strings, parsed. */
    val strings: Map<String, Any?>,
)

data class Verdict(
    val itemId: String,
    val name: String,
    val stream: String,
    val streamName: String,
    val binColour: String,
    val binColourName: String,
    val recyclable: Boolean,
    val compostable: Boolean,
    val steps: List<String>,
    val hazard: String?,
    val dropOff: String?,
    val dropOffText: String?,
    val notes: List<String>,
    val impact: String,
    val language: String,
    val city: String,
    val applied: List<String>,
    val missingTranslation: Boolean,
) {
    /** Same keys and order as Verdict.to_dict() in Python. */
    fun toMap(): Map<String, Any?> = linkedMapOf(
        "item_id" to itemId,
        "name" to name,
        "stream" to stream,
        "stream_name" to streamName,
        "bin_colour" to binColour,
        "bin_colour_name" to binColourName,
        "recyclable" to recyclable,
        "compostable" to compostable,
        "steps" to steps,
        "hazard" to hazard,
        "drop_off" to dropOff,
        "drop_off_text" to dropOffText,
        "notes" to notes,
        "impact" to impact,
        "language" to language,
        "city" to city,
        "applied" to applied,
        "missing_translation" to missingTranslation,
    )

    /** The facts handed to Llama, in the same key order as Python's Verdict.facts(). */
    fun facts(): Map<String, Any?> = linkedMapOf(
        "item" to name,
        "stream" to streamName,
        "bin" to binColourName,
        "recyclable" to recyclable,
        "steps" to steps,
        "hazard" to hazard,
        "drop_off" to dropOffText,
        "notes" to notes,
        "impact" to impact,
    )
}

/** Picks text in the requested language, falling back to English (Python's _Text). */
private class TextPicker(val lang: String) {
    var missing = false

    @Suppress("UNCHECKED_CAST")
    fun <T> pick(byLang: Any?): T? {
        if (byLang == null) return null
        val map = byLang as Map<String, Any?>
        if (lang in map) return map[lang] as T
        missing = true
        return map[FALLBACK_LANG] as T
    }
}

@Suppress("UNCHECKED_CAST")
class KnowledgeBase(private val data: KbData) {
    val version: String get() = data.version
    val strings: Map<String, Any?> get() = data.strings
    val itemIds: List<String> get() = data.items.keys.sorted()

    fun item(itemId: String): Map<String, Any?> = data.items.getValue(itemId)

    fun city(cityId: String): Map<String, Any?> = data.cities.getValue(cityId)

    fun itemForClass(detectorClass: String): String? = data.detectorClasses[detectorClass]

    /** Item ids whose name or alias matches the text: exact matches first, then partial. */
    fun search(text: String, lang: String? = null): List<String> {
        val query = text.trim().lowercase()
        if (query.isEmpty()) return emptyList()
        val rows = data.aliases.filter { lang == null || it.lang == lang }
        val exact = rows.filter { it.alias == query }.map { it.itemId }.toSortedSet()
        val partial = rows
            .filter { it.itemId !in exact && (query in it.alias || it.alias in query) }
            .map { it.itemId }
            .toSortedSet()
        return exact.toList() + partial.toList()
    }

    /** Decide stream, bin and instructions for an item in a given condition and city. */
    fun resolve(
        itemId: String,
        attributes: Map<String, Any?> = emptyMap(),
        language: String = "en",
        cityId: String = DEFAULT_CITY,
    ): Verdict {
        val item = data.items.getValue(itemId)
        val city = data.cities.getValue(cityId)
        val text = TextPicker(language)

        var stream = item["stream"] as String
        var recyclable = item["recyclable"] as Boolean
        var compostable = (item["compostable"] as Boolean?) ?: false
        var dropOff = item["drop_off"] as String?
        var hazard: Any? = item["hazard"]
        var stepsByLang: Any? = item["prep_steps"]
        val extraSteps = mutableListOf<String>()
        val notes = mutableListOf<String>()
        val applied = mutableListOf<String>()

        val override = (city["overrides"] as Map<String, Any?>? ?: emptyMap())[itemId]
            as Map<String, Any?>?
        if (override != null) {
            if ("stream" in override && override["stream"] != stream) {
                stream = override["stream"] as String
                dropOff = null
            }
            if ("recyclable" in override) recyclable = override["recyclable"] as Boolean
            if ("drop_off" in override) dropOff = override["drop_off"] as String?
            if ("note" in override) notes.add(text.pick<String>(override["note"])!!)
            applied.add("city:$cityId")
        }

        for (rule in item["condition_rules"] as List<Map<String, Any?>>? ?: emptyList()) {
            val condition = rule["if"] as Map<String, Any?>
            if (!condition.all { (key, value) -> attributes[key] == value }) continue
            val then = rule["then"] as Map<String, Any?>
            if ("stream" in then && then["stream"] != stream) {
                stream = then["stream"] as String
                dropOff = null
            }
            if ("recyclable" in then) recyclable = then["recyclable"] as Boolean
            if ("compostable" in then) compostable = then["compostable"] as Boolean
            if ("hazard" in then) hazard = then["hazard"]
            if ("drop_off" in then) dropOff = then["drop_off"] as String?
            if ("steps" in then) stepsByLang = then["steps"]
            if ("extra_steps" in then) extraSteps.addAll(text.pick<List<String>>(then["extra_steps"])!!)
            if ("note" in then) notes.add(text.pick<String>(then["note"])!!)
            applied.add("rule:" + condition.entries.joinToString(",") { (k, v) -> "$k=${pyStr(v).lowercase()}" })
        }

        val colour = (city["bin_colours"] as Map<String, Any?>)[stream] as String
        val dropOffText = dropOff?.let {
            text.pick<String>((city["drop_off_info"] as Map<String, Any?>)[it])
        }
        val streams = data.strings["streams"] as Map<String, Any?>
        val colours = data.strings["colours"] as Map<String, Any?>
        return Verdict(
            itemId = itemId,
            name = text.pick<String>(item["names"])!!,
            stream = stream,
            streamName = text.pick<String>(streams[stream])!!,
            binColour = colour,
            binColourName = text.pick<String>(colours[colour])!!,
            recyclable = recyclable,
            compostable = compostable,
            steps = extraSteps + text.pick<List<String>>(stepsByLang)!!,
            hazard = text.pick<String>(hazard),
            dropOff = dropOff,
            dropOffText = dropOffText,
            notes = notes.toList(),
            impact = text.pick<String>(item["impact"])!!,
            language = language,
            city = cityId,
            applied = applied.toList(),
            missingTranslation = text.missing,
        )
    }

    fun languageName(language: String): String =
        (data.strings["languages"] as Map<String, Any?>)[language] as String? ?: language

    /** A fixed app message (strings.json "messages"), falling back to English. */
    fun message(key: String, language: String): String {
        val byLang = (data.strings["messages"] as Map<String, Any?>)[key] as Map<String, Any?>
        return (byLang[language] ?: byLang.getValue(FALLBACK_LANG)) as String
    }

    /** Every name and alias of an item, in every language (same order as Python). */
    fun namesOf(itemId: String): List<String> {
        val item = data.items.getValue(itemId)
        val names = (item["names"] as Map<String, Any?>).values.map { it as String }.toMutableList()
        for (aliases in (item["aliases"] as Map<String, Any?>? ?: emptyMap()).values) {
            names.addAll(aliases as List<String>)
        }
        return names
    }
}

/** Python's str() for JSON scalars, used where the reference code calls str(value). */
internal fun pyStr(value: Any?): String = when (value) {
    null -> "None"
    true -> "True"
    false -> "False"
    else -> value.toString()
}
