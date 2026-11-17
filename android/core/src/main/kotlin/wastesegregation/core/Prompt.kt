package wastesegregation.core

/**
 * Llama 3.2 prompt builder: a port of pipeline/prompt.py. The text must match Python byte for
 * byte (the golden fixtures store it), because the system part is pre-filled into the KV
 * cache once and the fine-tune is trained on exactly this format.
 */
object Prompt {
    const val BOS = "<|begin_of_text|>"
    const val EOT = "<|eot_id|>"

    const val SYSTEM_PROMPT =
        "You are a waste disposal guide for India. Use only the FACTS given and never " +
            "contradict them. The app already shows the facts to the user; your job is to explain. " +
            "Reply with one JSON object with exactly two keys: \"why\" (answer the QUESTION in one or " +
            "two short sentences) and \"tip\" (one short practical tip). " +
            "Write both in the requested LANGUAGE. If the QUESTION is not about waste, say in why " +
            "that you can only help with waste. Output nothing except the JSON object."

    const val DEFAULT_QUESTION = "Explain how to dispose of this and why."

    /** The assistant turn starts with this text; the model continues inside "why". */
    const val ANSWER_PREFIX = "{\"why\": \""

    private fun header(role: String) = "<|start_header_id|>$role<|end_header_id|>\n\n"

    /** The fixed part of every prompt (BOS + system message), for KV cache pre-fill. */
    fun prefix(includeBos: Boolean = true): String =
        (if (includeBos) BOS else "") + header("system") + SYSTEM_PROMPT + EOT

    fun userMessage(
        verdict: Verdict,
        attributes: Map<String, Any?>,
        languageName: String,
        question: String? = null,
        short: Boolean = false,
    ): String {
        var facts = verdict.facts()
        if (short) facts = linkedMapOf<String, Any?>().apply {
            for (k in listOf("item", "stream", "bin", "steps")) put(k, facts[k])
        }
        val lines = mutableListOf("FACTS: ${Json.dumps(facts)}")
        if (!short) {
            val condition = linkedMapOf<String, Any?>()
            for (k in attributes.keys.sorted()) attributes[k]?.let { condition[k] = it }
            lines.add("CONDITION: ${Json.dumps(condition)}")
        }
        lines.add("LANGUAGE: $languageName")
        lines.add("QUESTION: ${if (question.isNullOrEmpty()) DEFAULT_QUESTION else question}")
        return lines.joinToString("\n")
    }

    /** Full prompt text, ending with the assistant header and ANSWER_PREFIX. */
    fun build(
        verdict: Verdict,
        attributes: Map<String, Any?>,
        languageName: String,
        question: String? = null,
        short: Boolean = false,
        includeBos: Boolean = true,
    ): String = prefix(includeBos) + header("user") +
        userMessage(verdict, attributes, languageName, question, short) + EOT +
        header("assistant") + ANSWER_PREFIX

    /** The complete answer text: ANSWER_PREFIX plus what the model generated after it. */
    fun fullAnswer(generated: String): String = ANSWER_PREFIX + generated
}
