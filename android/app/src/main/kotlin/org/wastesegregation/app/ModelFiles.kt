package org.wastesegregation.app

import android.content.Context
import java.io.File

/**
 * Where the app finds its files.
 *
 * - Small files ship in the APK assets (KB, detector .pte + class names, SigLIP text-side JSON)
 *   and are copied to internal storage once per install, because ExecuTorch loads by path.
 * - Large files are pushed with scripts/push_models.sh into the app's external files folder
 *   (/sdcard/Android/data/org.wastesegregation.app/files/models/): the Llama .pte + tokenizer
 *   and the SigLIP 2 image encoder. A detector pushed there overrides the bundled one.
 */
class ModelFiles(private val context: Context) {
    val pushedDir: File = (context.getExternalFilesDir("models") ?: File(context.filesDir, "models")).apply { mkdirs() }
    private val assetDir = File(context.filesDir, "bundled").apply { mkdirs() }
    private val installedAt: Long = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime
    private val assetNames: Set<String> = context.assets.list("")?.toSet() ?: emptySet()

    /** Copy a bundled asset to internal storage (once per install) and return its path. */
    fun asset(name: String): File? {
        if (name !in assetNames) return null
        val out = File(assetDir, name)
        if (!out.isFile || out.lastModified() < installedAt) {
            val tmp = File(assetDir, "$name.tmp")
            context.assets.open(name).use { input -> tmp.outputStream().use { input.copyTo(it) } }
            tmp.renameTo(out)
        }
        return out
    }

    fun pushed(name: String): File? = File(pushedDir, name).takeIf { it.isFile }

    /** The Llama .pte: [PREFERRED_LLAMA] if several were pushed, else the first by name. */
    fun llama(): File? {
        val all = pushedDir.listFiles()?.filter { it.name.startsWith("Llama") && it.name.endsWith(".pte") }.orEmpty()
        return all.firstOrNull { PREFERRED_LLAMA in it.name } ?: all.minByOrNull { it.name }
    }

    fun tokenizer(): File? = pushed("tokenizer.model")
    fun siglip(): File? = pushed("siglip2_image_int8.pte")
    fun asrModel(): File? = pushed("model.int8.onnx")
    fun asrTokens(): File? = pushed("tokens.txt")
    fun detector(): File? = pushed("model.pte") ?: asset("model.pte")
    fun detectorNames(): File? = pushed("metadata.yaml") ?: asset("metadata.yaml")
    fun kb(): File? = asset("kb.sqlite")
    fun textEmbeddings(): File? = asset("text_embeddings.json")
    fun attributePrompts(): File? = asset("attribute_prompts.json")

    companion object {
        /** Chosen by the answer-quality experiments (eval/results/llm_experiments). */
        const val PREFERRED_LLAMA = "SpinQuant"

        /** Class names in index order from an Ultralytics metadata.yaml ("names:" then "  0: x"). */
        fun classNames(yaml: File): List<String> {
            val names = sortedMapOf<Int, String>()
            var inNames = false
            for (line in yaml.readLines()) {
                if (line.startsWith("names:")) {
                    inNames = true
                    continue
                }
                if (inNames) {
                    val m = Regex("""^\s+(\d+):\s*['"]?([^'"]+?)['"]?\s*$""").find(line)
                    if (m != null) names[m.groupValues[1].toInt()] = m.groupValues[2] else if (!line.startsWith(" ")) break
                }
            }
            return names.values.toList()
        }
    }
}
