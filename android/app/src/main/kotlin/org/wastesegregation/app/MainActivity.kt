package org.wastesegregation.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import java.util.concurrent.Executor

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme { App() } }
    }
}

private val BIN_COLOURS = mapOf(
    "green" to Color(0xFF2E7D32),
    "blue" to Color(0xFF1565C0),
    "red" to Color(0xFFC62828),
    "black" to Color(0xFF212121),
)

@Composable
fun App(vm: AppViewModel = viewModel()) {
    var tab by remember { mutableStateOf(0) }
    val ui = vm.ui
    Scaffold(
        bottomBar = {
            NavigationBar {
                listOf("Scan", "Ask", "Impact", "Settings").forEachIndexed { i, label ->
                    NavigationBarItem(selected = tab == i, onClick = { tab = i }, icon = {}, label = { Text(label) })
                }
            }
        },
    ) { padding ->
        Box(Modifier.padding(padding).fillMaxSize()) {
            when (tab) {
                0 -> ScanScreen(vm)
                1 -> AskScreen(vm)
                2 -> ImpactScreen(vm)
                else -> SettingsScreen(vm)
            }
        }
    }
    ui.message?.let { msg ->
        AlertDialog(onDismissRequest = vm::clearMessage, confirmButton = {
            TextButton(onClick = vm::clearMessage) { Text("OK") }
        }, text = { Text(msg) })
    }
}

@Composable
fun ScanScreen(vm: AppViewModel) {
    val context = LocalContext.current
    val ui = vm.ui
    var granted by remember {
        mutableStateOf(context.checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let { context.contentResolver.openInputStream(it)?.use(BitmapFactory::decodeStream) }?.let(vm::photo)
    }
    var correcting by remember { mutableStateOf(false) }

    Column(Modifier.fillMaxSize()) {
        Box(Modifier.fillMaxWidth().weight(if (ui.verdict == null) 1f else 0.45f)) {
            if (granted) CameraPreview(vm) else {
                Button(onClick = { permission.launch(Manifest.permission.CAMERA) }, Modifier.align(Alignment.Center)) {
                    Text("Allow camera")
                }
            }
            Column(Modifier.align(Alignment.TopStart).background(Color(0x99000000)).padding(8.dp)) {
                val line = when (ui.scanState) {
                    "stable", "locked" -> ui.liveItem ?: ui.verdict?.name ?: ""
                    "unsure" -> "Not sure yet… hold steady"
                    else -> ui.status
                }
                Text(line, color = Color.White, fontWeight = FontWeight.Bold)
                if (ui.showOverlay) {
                    Text(
                        "detector ${ui.detectMs?.let { "%.0f ms".format(it) } ?: "–"} · card ${ui.cardMs?.let { "%.0f ms".format(it) } ?: "–"}" +
                            " · Llama ${ui.firstTokenS?.let { "%.1f s".format(it) } ?: "–"} first, ${ui.decodeTps?.let { "%.1f".format(it) } ?: "–"} tok/s",
                        color = Color.White, fontSize = 11.sp,
                    )
                }
            }
            Row(Modifier.align(Alignment.BottomCenter).padding(8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = { picker.launch("image/*") }) { Text("Photo", color = Color.White) }
                if (ui.verdict == null && ui.unsureFrames > 15) {
                    Button(onClick = { vm.suggest() }) { Text("Not sure? Suggest") }
                }
            }
        }
        if (ui.suggestions.isNotEmpty() && ui.verdict == null) {
            Card(Modifier.fillMaxWidth().padding(8.dp)) {
                Column(Modifier.padding(12.dp)) {
                    Text("Is it one of these?", fontWeight = FontWeight.Bold)
                    ui.suggestions.forEach { s ->
                        TextButton(onClick = { vm.choose(s.itemId) }) { Text("${vm.nameOf(s.itemId)} (${"%.2f".format(s.cosine)})") }
                    }
                    TextButton(onClick = { correcting = true }) { Text("None of these: pick from the list") }
                }
            }
        }
        ui.verdict?.let { VerdictCard(vm, Modifier.weight(0.55f), onWrong = { correcting = true }) }
    }
    if (correcting) {
        ItemPicker(vm, onPick = {
            correcting = false
            if (ui.verdict != null) vm.correct(it) else vm.choose(it)
        }, onDismiss = { correcting = false })
    }
}

@Composable
fun CameraPreview(vm: AppViewModel) {
    val lifecycleOwner = LocalLifecycleOwner.current
    AndroidView(modifier = Modifier.fillMaxSize(), factory = { ctx ->
        PreviewView(ctx).also { view ->
            val mainExecutor = Executor { Handler(Looper.getMainLooper()).post(it) }
            val future = ProcessCameraProvider.getInstance(ctx)
            future.addListener({
                val provider = future.get()
                val preview = Preview.Builder().build().also { it.setSurfaceProvider(view.surfaceProvider) }
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                analysis.setAnalyzer(vm.analysisExecutor, vm::onFrame)
                provider.unbindAll()
                provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            }, mainExecutor)
        }
    })
}

@Composable
fun VerdictCard(vm: AppViewModel, modifier: Modifier, onWrong: () -> Unit) {
    val ui = vm.ui
    val v = ui.verdict ?: return
    val hi = v.language == "hi"
    Column(modifier.fillMaxWidth().verticalScroll(rememberScrollState())) {
        Box(Modifier.fillMaxWidth().background(BIN_COLOURS[v.binColour] ?: Color.Gray).padding(16.dp)) {
            Column {
                Text(v.name, color = Color.White, fontSize = 22.sp, fontWeight = FontWeight.Bold)
                Text(
                    if (hi) "${v.binColourName} डिब्बा · ${v.streamName}" else "${v.binColourName} bin · ${v.streamName}",
                    color = Color.White, fontSize = 18.sp,
                )
            }
        }
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            ui.askAttributes.forEach { attr ->
                val q = when (attr) {
                    "broken" -> if (hi) "क्या यह टूटा है?" else "Is it broken?"
                    else -> if (hi) "क्या यह गंदा है (खाना/तेल लगा)?" else "Is it dirty (food or oil on it)?"
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(q, Modifier.weight(1f))
                    TextButton(onClick = { vm.setAttribute(attr, true) }) { Text(if (hi) "हाँ" else "Yes") }
                    TextButton(onClick = { vm.setAttribute(attr, false) }) { Text(if (hi) "नहीं" else "No") }
                }
            }
            v.steps.forEachIndexed { i, s -> Text("${i + 1}. $s") }
            v.hazard?.let { Text("⚠ $it", color = Color(0xFFC62828), fontWeight = FontWeight.Bold) }
            v.dropOffText?.let { Text(it) }
            v.notes.forEach { Text(it, fontSize = 13.sp) }
            Spacer(Modifier.height(4.dp))
            ui.question?.let { Text("Q: $it", fontWeight = FontWeight.Bold) }
            val verified = ui.verified
            when {
                verified != null -> {
                    verified.output["why"]?.let { Text(it as String) }
                    verified.output["tip"]?.let { Text("Tip: $it", fontSize = 13.sp) }
                    if (verified.status != "ok") {
                        Text("(${verified.status}: shown from the knowledge base)", fontSize = 11.sp, color = Color.Gray)
                    }
                }
                ui.explaining -> Text(ui.whyStream.ifEmpty { "Explaining…" }, color = Color.DarkGray)
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = vm::scanAgain) { Text(if (hi) "फिर से स्कैन" else "Scan again") }
                OutlinedButton(onClick = onWrong) { Text(if (hi) "यह गलत है" else "This is wrong") }
            }
        }
    }
}

@Composable
fun ItemPicker(vm: AppViewModel, onPick: (String) -> Unit, onDismiss: () -> Unit) {
    var filter by remember { mutableStateOf("") }
    val all = remember { vm.itemChoices() }
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
        title = { Text("What is it?") },
        text = {
            Column {
                OutlinedTextField(filter, { filter = it }, label = { Text("Search") }, singleLine = true)
                LazyColumn(Modifier.height(320.dp)) {
                    items(all.filter { it.second.contains(filter, ignoreCase = true) }) { (id, name) ->
                        TextButton(onClick = { onPick(id) }) { Text(name) }
                    }
                }
            }
        },
    )
}

@Composable
fun AskScreen(vm: AppViewModel) {
    val ui = vm.ui
    var text by remember { mutableStateOf("") }
    val hi = ui.language == "hi"
    Column(Modifier.fillMaxSize().padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            ui.verdict?.let { (if (hi) "विषय: " else "About: ") + it.name }
                ?: if (hi) "पहले कोई चीज़ स्कैन करें, या सवाल में उसका नाम लिखें।" else "Scan an item first, or name it in your question.",
            fontWeight = FontWeight.Bold,
        )
        OutlinedTextField(text, { text = it }, Modifier.fillMaxWidth(), label = { Text(if (hi) "आपका सवाल" else "Your question") })
        Button(onClick = { if (text.isNotBlank()) vm.ask(text.trim()) }, enabled = !ui.explaining) {
            Text(if (hi) "पूछें" else "Ask")
        }
        if (ui.verdict != null) VerdictCard(vm, Modifier.weight(1f), onWrong = {})
    }
}

@Composable
fun ImpactScreen(vm: AppViewModel) {
    val ui = vm.ui
    LazyColumn(Modifier.fillMaxSize().padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item {
            val s = ui.impact
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp)) {
                    Text("Your sorting", fontWeight = FontWeight.Bold, fontSize = 18.sp)
                    if (s == null) Text("No scans yet") else {
                        Text("Items sorted: ${s["scans"]}")
                        Text("By stream: ${s["by_stream"]}")
                        Text("Kept out of landfill (estimate): ${s["diverted_items"]} items, ~${s["diverted_kg_estimate"]} kg")
                        Text("Needed cleaning: ${s["needed_cleaning_rate"] ?: "–"}")
                        Text("Streak: ${s["current_streak_days"]} days")
                        Text("Estimates use typical item weights from the knowledge base.", fontSize = 11.sp, color = Color.Gray)
                    }
                }
            }
        }
        ui.didYouKnow?.let { d ->
            item {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(12.dp)) {
                        Text("${d["heading"]}: ${d["title"]}", fontWeight = FontWeight.Bold)
                        Text(d["text"] as String)
                    }
                }
            }
        }
        ui.quiz?.let { q -> item { QuizCard(q) } }
    }
}

@Suppress("UNCHECKED_CAST")
@Composable
fun QuizCard(q: Map<String, Any?>) {
    var picked by remember(q) { mutableStateOf<String?>(null) }
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp)) {
            Text(q["question"] as String, fontWeight = FontWeight.Bold)
            (q["options"] as List<Map<String, Any?>>).forEach { o ->
                TextButton(onClick = { picked = o["stream"] as String }) { Text(o["label"] as String) }
            }
            picked?.let {
                Text(if (it == q["answer"]) "Correct! " + q["explanation"] else "Not quite. " + q["explanation"])
            }
        }
    }
}

@Composable
fun SettingsScreen(vm: AppViewModel) {
    val ui = vm.ui
    var exported by remember { mutableStateOf<String?>(null) }
    Column(Modifier.fillMaxSize().padding(12.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("Language", fontWeight = FontWeight.Bold)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf("en" to "English", "hi" to "हिंदी").forEach { (code, label) ->
                if (ui.language == code) Button(onClick = {}) { Text(label) }
                else OutlinedButton(onClick = { vm.setLanguage(code) }) { Text(label) }
            }
        }
        Text("Models (all on this phone, no internet)", fontWeight = FontWeight.Bold)
        ui.models.forEach { (name, status) -> Text("$name: $status", fontSize = 13.sp) }
        OutlinedButton(onClick = vm::toggleOverlay) { Text(if (ui.showOverlay) "Hide timing overlay" else "Show timing overlay") }
        Button(onClick = { exported = vm.exportTimings() }) { Text("Export timings for the report") }
        exported?.let { Text("Saved: $it", fontSize = 12.sp) }
        Text(
            "Built with Llama. Llama 3.2 is licensed under the Llama 3.2 Community License, " +
                "Copyright © Meta Platforms, Inc. All Rights Reserved.",
            fontSize = 11.sp, color = Color.Gray,
        )
    }
}
