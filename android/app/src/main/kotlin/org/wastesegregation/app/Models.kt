package org.wastesegregation.app

import android.database.sqlite.SQLiteDatabase
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.pytorch.executorch.extension.llm.LlmModule
import wastesegregation.core.AliasRow
import wastesegregation.core.Detect
import wastesegregation.core.Detection
import wastesegregation.core.FallbackRank
import wastesegregation.core.Json
import wastesegregation.core.JsonNumber
import wastesegregation.core.KbData
import wastesegregation.core.KnowledgeBase
import wastesegregation.core.LatencyLog
import wastesegregation.core.Stages
import wastesegregation.core.Suggestion
import java.io.Closeable
import java.io.File

/** Reads kb.sqlite (built by kb/build_kb.py) into the core's KnowledgeBase. */
@Suppress("UNCHECKED_CAST")
object KbLoader {
    fun load(file: File): KnowledgeBase =
        SQLiteDatabase.openDatabase(file.path, null, SQLiteDatabase.OPEN_READONLY).use { db ->
            fun rows(sql: String): List<List<String>> = db.rawQuery(sql, null).use { c ->
                buildList { while (c.moveToNext()) add((0 until c.columnCount).map { c.getString(it) }) }
            }
            fun obj(json: String) = Json.parse(json) as Map<String, Any?>
            val meta = rows("SELECT key, value FROM meta").associate { it[0] to it[1] }
            KnowledgeBase(
                KbData(
                    items = rows("SELECT id, json FROM items").associate { it[0] to obj(it[1]) },
                    cities = rows("SELECT id, json FROM cities").associate { it[0] to obj(it[1]) },
                    aliases = rows("SELECT alias, lang, item_id FROM aliases").map { AliasRow(it[0], it[1], it[2]) },
                    detectorClasses = rows("SELECT class, item_id FROM detector_classes").associate { it[0] to it[1] },
                    version = meta.getValue("kb_version"),
                    strings = obj(meta.getValue("strings")),
                ),
            )
        }
}

/** YOLO26n .pte: letterbox (same geometry as pipeline/detect.py) -> forward -> core decode + NMS. */
class Detector(path: File, private val classNames: List<String>, private val kb: KnowledgeBase) : Closeable {
    private val module = Module.load(path.path)
    private val size = Detect.INPUT_SIZE
    private val canvasBitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(canvasBitmap)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val pixels = IntArray(size * size)
    private val input = FloatArray(3 * size * size)
    private val shape = longArrayOf(1, 3, size.toLong(), size.toLong())

    fun detect(src: Bitmap, log: LatencyLog): List<Detection> {
        val geo = log.time(Stages.PREPROCESS) {
            val g = Detect.letterboxGeometry(src.width, src.height, size)
            canvas.drawColor(Color.rgb(Detect.PAD_VALUE, Detect.PAD_VALUE, Detect.PAD_VALUE))
            val lb = g.letterbox
            canvas.drawBitmap(src, null, Rect(lb.left, lb.top, lb.left + g.newW, lb.top + g.newH), paint)
            canvasBitmap.getPixels(pixels, 0, size, 0, 0, size, size)
            val plane = size * size
            for (i in 0 until plane) {
                val p = pixels[i]
                input[i] = ((p shr 16) and 0xFF) / 255f
                input[plane + i] = ((p shr 8) and 0xFF) / 255f
                input[2 * plane + i] = (p and 0xFF) / 255f
            }
            g
        }
        val out = log.time(Stages.DETECT) {
            module.forward(EValue.from(Tensor.fromBlob(input, shape)))[0].toTensor()
        }
        val outShape = out.shape()
        val data = out.dataAsFloatArray
        val rows = outShape[1].toInt()
        val anchors = outShape[2].toInt()
        val output = Array(rows) { r -> data.copyOfRange(r * anchors, (r + 1) * anchors) }
        return Detect.decode(output, geo.letterbox, classNames, knownItems = classNames.toSet())
            .map { it.copy(itemId = kb.itemForClass(it.detectorClass)) }
    }

    override fun close() = module.destroy()
}

/** SigLIP 2 image encoder (int8 .pte) for the unknown-item fallback and the dirty/broken check. */
@Suppress("UNCHECKED_CAST")
class Siglip(path: File, textEmbeddings: File) : Closeable {
    private val module = Module.load(path.path)
    private val itemIds: List<String>
    private val text: List<FloatArray>
    private val scale: Double
    private val bias: Double
    private val input = FloatArray(3 * SIZE * SIZE)
    private val pixels = IntArray(SIZE * SIZE)

    init {
        val data = Json.parse(textEmbeddings.readText()) as Map<String, Any?>
        itemIds = (data["item_ids"] as List<Any?>).map { it as String }
        text = (data["embeddings"] as List<List<Any?>>).map { row ->
            FloatArray(row.size) { (row[it] as JsonNumber).text.toFloat() }
        }
        scale = (data["logit_scale"] as JsonNumber).text.toDouble()
        bias = (data["logit_bias"] as JsonNumber).text.toDouble()
    }

    /** RGB 224x224 bilinear (no padding), [0,1], then (x - 0.5) / 0.5; L2-normalised output. */
    fun embed(bitmap: Bitmap): FloatArray {
        val scaled = Bitmap.createScaledBitmap(bitmap, SIZE, SIZE, true)
        scaled.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val plane = SIZE * SIZE
        for (i in 0 until plane) {
            val p = pixels[i]
            input[i] = (((p shr 16) and 0xFF) / 255f - 0.5f) / 0.5f
            input[plane + i] = (((p shr 8) and 0xFF) / 255f - 0.5f) / 0.5f
            input[2 * plane + i] = ((p and 0xFF) / 255f - 0.5f) / 0.5f
        }
        val shape = longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())
        return module.forward(EValue.from(Tensor.fromBlob(input, shape)))[0].toTensor().dataAsFloatArray
    }

    fun suggest(embedding: FloatArray, k: Int = 3): List<Suggestion> =
        FallbackRank.rank(text, itemIds, embedding, scale, bias, k)

    override fun close() = module.destroy()

    companion object {
        const val SIZE = 224
    }
}

/**
 * Llama 3.2 through ExecuTorch's LlmModule. The prompt from the core's Prompt already starts
 * with <|begin_of_text|>, so the runner adds no BOS (numBos 0), as on the laptop.
 */
class Llama(model: File, tokenizer: File) : Closeable {
    private val module = LlmModule(model.path, tokenizer.path, 0f)

    /** The runner's last stats JSON (prompt/generated tokens, timings), for the benchmark export. */
    @Volatile var lastStats: String? = null
        private set

    fun load() = module.load()

    /** Generate after [prompt]; [onToken] gets each piece as it arrives. Returns all the text. */
    fun generate(prompt: String, maxNewTokens: Int = 200, temperature: Float = 0f, onToken: (String) -> Unit): String {
        val out = StringBuilder()
        val config = LlmGenerationConfig.create()
            .echo(false)
            .maxNewTokens(maxNewTokens)
            .temperature(temperature)
            .numBos(0)
            .numEos(0)
            .build()
        module.resetContext()
        module.generate(
            prompt,
            config,
            object : LlmCallback {
                override fun onResult(result: String) {
                    out.append(result)
                    onToken(result)
                }

                override fun onStats(stats: String) {
                    lastStats = stats
                }
            },
        )
        return out.toString()
    }

    fun stop() = module.stop()

    override fun close() = module.close()
}
