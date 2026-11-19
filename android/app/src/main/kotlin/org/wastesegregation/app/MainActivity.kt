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
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import java.util.concurrent.Executor

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContent { AppTheme { App() } }
    }
}

private val TABS = listOf(
    Triple("Scan", "स्कैन", AppIcon.Scan),
    Triple("Ask", "पूछें", AppIcon.Ask),
    Triple("Impact", "असर", AppIcon.Impact),
    Triple("Settings", "सेटिंग", AppIcon.Settings),
)

@Composable
fun App(vm: AppViewModel = viewModel()) {
    var tab by remember { mutableStateOf(0) }
    val ui = vm.ui
    val hi = ui.language == "hi"
    Scaffold(
        containerColor = Palette.Background,
        bottomBar = {
            Column {
                HorizontalDivider(color = Palette.Outline)
                NavigationBar(containerColor = Palette.Background, tonalElevation = 0.dp) {
                    TABS.forEachIndexed { i, (en, hindi, icon) ->
                        val selected = tab == i
                        NavigationBarItem(
                            selected = selected,
                            onClick = { tab = i },
                            icon = { Icon(icon, if (selected) Palette.Accent else Palette.InkMuted) },
                            label = { Text(if (hi) hindi else en, style = MaterialTheme.typography.labelMedium) },
                            colors = NavigationBarItemDefaults.colors(
                                indicatorColor = Palette.AccentSoft,
                                selectedTextColor = Palette.Accent,
                                unselectedTextColor = Palette.InkMuted,
                            ),
                        )
                    }
                }
            }
        },
    ) { padding ->
        Box(Modifier.padding(padding).fillMaxSize().background(Palette.Background)) {
            when (tab) {
                0 -> ScanScreen(vm)
                1 -> AskScreen(vm)
                2 -> ImpactScreen(vm)
                else -> SettingsScreen(vm)
            }
        }
    }
    ui.message?.let { msg ->
        AlertDialog(
            onDismissRequest = vm::clearMessage,
            containerColor = Palette.Background,
            confirmButton = { TextButton(onClick = vm::clearMessage) { Text("OK", color = Palette.Accent) } },
            text = { Text(msg, color = Palette.Ink) },
        )
    }
}

// ---------------------------------------------------------------------------------------- shared

@Composable
fun Panel(modifier: Modifier = Modifier, padding: Dp = 16.dp, content: @Composable () -> Unit) {
    Box(
        modifier
            .fillMaxWidth()
            .clip(MaterialTheme.shapes.medium)
            .background(Palette.Surface)
            .padding(padding),
    ) { content() }
}

@Composable
fun SectionLabel(text: String) {
    Text(text.uppercase(), style = MaterialTheme.typography.labelMedium, color = Palette.InkMuted)
}

@Composable
fun BinChip(colour: Color, text: String, onDark: Boolean = false) {
    Row(
        Modifier
            .clip(RoundedCornerShape(50))
            .background(if (onDark) Color.White.copy(alpha = 0.18f) else colour.copy(alpha = 0.12f))
            .padding(horizontal = 10.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.size(8.dp).clip(CircleShape).background(if (onDark) Color.White else colour))
        Spacer(Modifier.width(6.dp))
        Text(text, style = MaterialTheme.typography.labelMedium, color = if (onDark) Color.White else colour)
    }
}

@Composable
fun PrimaryButton(text: String, modifier: Modifier = Modifier, enabled: Boolean = true, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        enabled = enabled,
        modifier = modifier.height(48.dp),
        shape = MaterialTheme.shapes.small,
        colors = ButtonDefaults.buttonColors(containerColor = Palette.Accent, contentColor = Color.White),
    ) { Text(text, style = MaterialTheme.typography.labelLarge) }
}

@Composable
fun SecondaryButton(text: String, modifier: Modifier = Modifier, enabled: Boolean = true, onClick: () -> Unit) {
    OutlinedButton(
        onClick = onClick,
        enabled = enabled,
        modifier = modifier.height(48.dp),
        shape = MaterialTheme.shapes.small,
        border = androidx.compose.foundation.BorderStroke(1.dp, Palette.Outline),
        colors = ButtonDefaults.outlinedButtonColors(contentColor = Palette.Ink),
    ) { Text(text, style = MaterialTheme.typography.labelLarge) }
}

// ------------------------------------------------------------------------------------------ scan

@Composable
fun ScanScreen(vm: AppViewModel) {
    val context = LocalContext.current
    val ui = vm.ui
    val hi = ui.language == "hi"
    var granted by remember {
        mutableStateOf(context.checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let { context.contentResolver.openInputStream(it)?.use(BitmapFactory::decodeStream) }?.let(vm::photo)
    }
    var correcting by remember { mutableStateOf(false) }
    val choosing = ui.verdict == null && (ui.suggestions.isNotEmpty() || ui.lookingCloser)

    Column(Modifier.fillMaxSize().padding(horizontal = 16.dp, vertical = 12.dp)) {
        // Camera: large while searching, a strip once there is a verdict or a choice to make.
        Box(
            Modifier
                .fillMaxWidth()
                .weight(if (ui.verdict == null && !choosing) 1f else 0.38f)
                .clip(MaterialTheme.shapes.large)
                .background(Palette.Ink),
        ) {
            if (granted) CameraPreview(vm) else {
                Column(Modifier.align(Alignment.Center), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        if (hi) "स्कैन करने के लिए कैमरा चाहिए" else "The camera is needed to scan",
                        color = Color.White, style = MaterialTheme.typography.bodyMedium,
                    )
                    Spacer(Modifier.height(12.dp))
                    PrimaryButton(if (hi) "कैमरा चालू करें" else "Allow camera") {
                        permission.launch(Manifest.permission.CAMERA)
                    }
                }
            }
            if (granted && ui.verdict == null) Viewfinder(Modifier.align(Alignment.Center))
            StatusPill(ui, hi, Modifier.align(Alignment.TopCenter).padding(top = 12.dp))
            Row(
                Modifier.align(Alignment.BottomCenter).padding(12.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                CameraAction(AppIcon.Photo, if (hi) "फ़ोटो" else "Photo") { picker.launch("image/*") }
                if (ui.verdict == null && ui.canSuggest && !choosing) {
                    CameraAction(AppIcon.Scan, if (hi) "सुझाव दें" else "Suggest") { vm.suggest() }
                }
            }
        }
        Spacer(Modifier.height(12.dp))
        when {
            ui.verdict != null -> VerdictCard(vm, Modifier.weight(0.62f), onWrong = { correcting = true })
            choosing -> SuggestionCard(vm, Modifier.weight(0.62f), onOther = { correcting = true })
            else -> Hint(ui, hi)
        }
    }
    if (correcting) {
        ItemPicker(vm, onPick = {
            correcting = false
            if (ui.verdict != null) vm.correct(it) else vm.choose(it)
        }, onDismiss = { correcting = false })
    }
}

@Composable
private fun Hint(ui: UiState, hi: Boolean) {
    Text(
        if (hi) "एक चीज़ कैमरे के सामने रखें। जवाब सबसे पहले, फिर उसका कारण।"
        else "Hold one item in view. The bin appears first, then why.",
        style = MaterialTheme.typography.bodyMedium,
        color = Palette.InkMuted,
        modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp),
        textAlign = TextAlign.Center,
    )
    if (ui.showOverlay) TimingLine(ui, Modifier.fillMaxWidth().padding(top = 6.dp))
}

@Composable
private fun TimingLine(ui: UiState, modifier: Modifier = Modifier) {
    Text(
        "detector ${ui.detectMs?.let { "%.0f ms".format(it) } ?: "–"} · card ${ui.cardMs?.let { "%.0f ms".format(it) } ?: "–"}" +
            " · Llama ${ui.firstTokenS?.let { "%.1f s".format(it) } ?: "–"} first, ${ui.decodeTps?.let { "%.1f".format(it) } ?: "–"} tok/s",
        style = MaterialTheme.typography.labelSmall,
        color = Palette.InkMuted,
        textAlign = TextAlign.Center,
        modifier = modifier,
    )
}

@Composable
private fun Viewfinder(modifier: Modifier) {
    Canvas(modifier.size(200.dp)) {
        val c = Color.White.copy(alpha = 0.85f)
        val w = size.width
        val l = w * 0.16f
        val sw = 3.dp.toPx()
        for ((x, y, dx, dy) in listOf(
            listOf(0f, 0f, 1f, 1f), listOf(w, 0f, -1f, 1f), listOf(0f, w, 1f, -1f), listOf(w, w, -1f, -1f),
        )) {
            drawLine(c, Offset(x, y), Offset(x + l * dx, y), sw, StrokeCap.Round)
            drawLine(c, Offset(x, y), Offset(x, y + l * dy), sw, StrokeCap.Round)
        }
    }
}

@Composable
private fun StatusPill(ui: UiState, hi: Boolean, modifier: Modifier) {
    val (dot, text) = when {
        ui.lookingCloser -> Palette.Amber to (if (hi) "ध्यान से देख रहे हैं…" else "Looking closer…")
        ui.verdict != null -> Bins.colour(ui.verdict.binColour) to ui.verdict.name
        ui.suggestions.isNotEmpty() -> Palette.Amber to (if (hi) "पक्का नहीं: चुनें" else "Not sure: pick one below")
        ui.scanState == "stable" -> Palette.Accent to (ui.liveItem ?: "")
        ui.scanState == "unsure" -> Palette.Amber to (if (hi) "पक्का नहीं… स्थिर रखें" else "Not sure yet… hold steady")
        ui.status.startsWith("Point") -> Palette.InkMuted to (if (hi) "एक चीज़ की ओर कैमरा करें" else "Looking for an item")
        else -> Palette.InkMuted to ui.status
    }
    if (text.isEmpty()) return
    Row(
        modifier
            .clip(RoundedCornerShape(50))
            .background(Color.White.copy(alpha = 0.94f))
            .padding(horizontal = 14.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (ui.lookingCloser) {
            CircularProgressIndicator(Modifier.size(12.dp), color = dot, strokeWidth = 2.dp)
        } else {
            Box(Modifier.size(8.dp).clip(CircleShape).background(dot))
        }
        Spacer(Modifier.width(8.dp))
        Text(text, style = MaterialTheme.typography.labelLarge, color = Palette.Ink)
    }
}

@Composable
private fun CameraAction(icon: AppIcon, label: String, onClick: () -> Unit) {
    Row(
        Modifier
            .clip(RoundedCornerShape(50))
            .background(Color.White.copy(alpha = 0.94f))
            .clickable(onClick = onClick)
            .padding(horizontal = 14.dp, vertical = 9.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(icon, Palette.Ink, 18.dp)
        Spacer(Modifier.width(6.dp))
        Text(label, style = MaterialTheme.typography.labelLarge, color = Palette.Ink)
    }
}

@Composable
fun CameraPreview(vm: AppViewModel) {
    val lifecycleOwner = LocalLifecycleOwner.current
    AndroidView(modifier = Modifier.fillMaxSize(), factory = { ctx ->
        PreviewView(ctx).also { view ->
            view.scaleType = PreviewView.ScaleType.FILL_CENTER
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
fun SuggestionCard(vm: AppViewModel, modifier: Modifier, onOther: () -> Unit) {
    val ui = vm.ui
    val hi = ui.language == "hi"
    Column(modifier.fillMaxWidth().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text(if (hi) "क्या यह इनमें से कोई है?" else "Is it one of these?", style = MaterialTheme.typography.titleLarge)
        if (ui.lookingCloser && ui.suggestions.isEmpty()) {
            Panel {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(Modifier.size(18.dp), color = Palette.Accent, strokeWidth = 2.dp)
                    Spacer(Modifier.width(12.dp))
                    Text(
                        if (hi) "मिलती-जुलती चीज़ें ढूँढ रहे हैं…" else "Finding the closest matches…",
                        style = MaterialTheme.typography.bodyMedium, color = Palette.InkMuted,
                    )
                }
            }
        }
        ui.suggestions.forEach { s ->
            val bin = vm.binColourOf(s.itemId)
            Row(
                Modifier
                    .fillMaxWidth()
                    .clip(MaterialTheme.shapes.medium)
                    .border(1.dp, Palette.Outline, MaterialTheme.shapes.medium)
                    .clickable { vm.choose(s.itemId) }
                    .padding(horizontal = 16.dp, vertical = 14.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Box(Modifier.size(12.dp).clip(CircleShape).background(Bins.colour(bin)))
                Spacer(Modifier.width(12.dp))
                Text(vm.nameOf(s.itemId), style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
                Text("›", style = MaterialTheme.typography.titleLarge, color = Palette.InkMuted)
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            SecondaryButton(if (hi) "कोई और" else "Something else", Modifier.weight(1f), onClick = onOther)
            SecondaryButton(if (hi) "फिर से स्कैन" else "Scan again", Modifier.weight(1f), onClick = vm::scanAgain)
        }
    }
}

@Composable
fun VerdictCard(vm: AppViewModel, modifier: Modifier, onWrong: () -> Unit) {
    val ui = vm.ui
    val v = ui.verdict ?: return
    val hi = v.language == "hi"
    val binColour = Bins.colour(v.binColour)
    Column(modifier.fillMaxWidth().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        // Header: the bin colour carries the answer.
        Column(
            Modifier
                .fillMaxWidth()
                .clip(MaterialTheme.shapes.medium)
                .background(binColour)
                .padding(16.dp),
        ) {
            Text(v.name, style = MaterialTheme.typography.headlineSmall, color = Color.White)
            Spacer(Modifier.height(8.dp))
            BinChip(
                Color.White,
                if (hi) "${v.binColourName} डिब्बा · ${v.streamName}" else "${v.binColourName} bin · ${v.streamName}",
                onDark = true,
            )
        }
        ui.askAttributes.forEach { attr ->
            val q = when (attr) {
                "broken" -> if (hi) "क्या यह टूटा है?" else "Is it broken?"
                else -> if (hi) "क्या यह गंदा है (खाना/तेल लगा)?" else "Is it dirty (food or oil on it)?"
            }
            Panel(padding = 12.dp) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(q, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
                    TextButton(onClick = { vm.setAttribute(attr, true) }) { Text(if (hi) "हाँ" else "Yes", color = Palette.Accent) }
                    TextButton(onClick = { vm.setAttribute(attr, false) }) { Text(if (hi) "नहीं" else "No", color = Palette.Accent) }
                }
            }
        }
        v.hazard?.let {
            Box(
                Modifier.fillMaxWidth().clip(MaterialTheme.shapes.small).background(Palette.WarningSoft).padding(12.dp),
            ) { Text(it, style = MaterialTheme.typography.bodyMedium, color = Palette.Warning, fontWeight = FontWeight.Medium) }
        }
        if (v.steps.isNotEmpty()) {
            SectionLabel(if (hi) "क्या करें" else "What to do")
            v.steps.forEachIndexed { i, s ->
                Row(verticalAlignment = Alignment.Top) {
                    Box(
                        Modifier.size(24.dp).clip(CircleShape).background(binColour.copy(alpha = 0.12f)),
                        contentAlignment = Alignment.Center,
                    ) { Text("${i + 1}", style = MaterialTheme.typography.labelMedium, color = binColour) }
                    Spacer(Modifier.width(10.dp))
                    Text(s, style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(top = 1.dp))
                }
            }
        }
        v.dropOffText?.let { Text(it, style = MaterialTheme.typography.bodyMedium, color = Palette.InkMuted) }
        v.notes.forEach { Text(it, style = MaterialTheme.typography.bodyMedium, color = Palette.InkMuted) }

        SectionLabel(if (hi) "क्यों" else "Why")
        Panel {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                ui.question?.let { Text(it, style = MaterialTheme.typography.titleMedium) }
                val verified = ui.verified
                when {
                    verified != null -> {
                        verified.output["why"]?.let { Text(it as String, style = MaterialTheme.typography.bodyLarge) }
                        verified.output["tip"]?.let {
                            Text(
                                (if (hi) "सुझाव: " else "Tip: ") + it,
                                style = MaterialTheme.typography.bodyMedium, color = Palette.Accent,
                            )
                        }
                        if (verified.status != "ok") {
                            Text(
                                if (hi) "ज्ञान-आधार से दिखाया गया" else "Shown from the knowledge base",
                                style = MaterialTheme.typography.labelSmall, color = Palette.InkMuted,
                            )
                        }
                    }
                    ui.explaining -> Row(verticalAlignment = Alignment.Top) {
                        CircularProgressIndicator(Modifier.size(14.dp).padding(top = 3.dp), color = Palette.Accent, strokeWidth = 2.dp)
                        Spacer(Modifier.width(10.dp))
                        Text(
                            ui.whyStream.ifEmpty { if (hi) "Llama समझा रहा है…" else "Llama is explaining…" },
                            style = MaterialTheme.typography.bodyLarge,
                            fontStyle = if (ui.whyStream.isEmpty()) FontStyle.Italic else FontStyle.Normal,
                            color = if (ui.whyStream.isEmpty()) Palette.InkMuted else Palette.Ink,
                        )
                    }
                    else -> Text("–", color = Palette.InkMuted)
                }
            }
        }
        if (ui.showOverlay) TimingLine(ui, Modifier.fillMaxWidth())
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            PrimaryButton(if (hi) "फिर से स्कैन" else "Scan again", Modifier.weight(1f), onClick = vm::scanAgain)
            SecondaryButton(if (hi) "यह गलत है" else "This is wrong", Modifier.weight(1f), onClick = onWrong)
        }
        Spacer(Modifier.height(4.dp))
    }
}

@Composable
fun ItemPicker(vm: AppViewModel, onPick: (String) -> Unit, onDismiss: () -> Unit) {
    var filter by remember { mutableStateOf("") }
    val all = remember { vm.itemChoices() }
    val hi = vm.ui.language == "hi"
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = Palette.Background,
        confirmButton = { TextButton(onClick = onDismiss) { Text(if (hi) "रद्द करें" else "Cancel", color = Palette.Accent) } },
        title = { Text(if (hi) "यह क्या है?" else "What is it?", style = MaterialTheme.typography.titleLarge) },
        text = {
            Column {
                OutlinedTextField(
                    filter, { filter = it },
                    label = { Text(if (hi) "खोजें" else "Search") }, singleLine = true,
                    shape = MaterialTheme.shapes.small,
                    colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = Palette.Accent, unfocusedBorderColor = Palette.Outline),
                    modifier = Modifier.fillMaxWidth(),
                )
                Spacer(Modifier.height(8.dp))
                LazyColumn(Modifier.height(340.dp)) {
                    items(all.filter { it.second.contains(filter, ignoreCase = true) }) { (id, name) ->
                        Row(
                            Modifier.fillMaxWidth().clickable { onPick(id) }.padding(vertical = 12.dp, horizontal = 4.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Box(Modifier.size(10.dp).clip(CircleShape).background(Bins.colour(vm.binColourOf(id))))
                            Spacer(Modifier.width(12.dp))
                            Text(name, style = MaterialTheme.typography.bodyLarge)
                        }
                        HorizontalDivider(color = Palette.Outline)
                    }
                }
            }
        },
    )
}

// ------------------------------------------------------------------------------------------- ask

@Composable
fun AskScreen(vm: AppViewModel) {
    val ui = vm.ui
    val context = LocalContext.current
    var text by remember { mutableStateOf("") }
    val hi = ui.language == "hi"
    val mic = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) vm.startVoice()
    }
    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(if (hi) "पूछें" else "Ask", style = MaterialTheme.typography.headlineSmall)
        Text(
            ui.verdict?.let { (if (hi) "अभी: " else "About: ") + it.name }
                ?: if (hi) "पहले कोई चीज़ स्कैन करें, या सवाल में उसका नाम लिखें।" else "Scan an item first, or name it in your question.",
            style = MaterialTheme.typography.bodyMedium, color = Palette.InkMuted,
        )
        OutlinedTextField(
            text, { text = it }, Modifier.fillMaxWidth(),
            label = { Text(if (hi) "आपका सवाल" else "Your question") },
            shape = MaterialTheme.shapes.small,
            colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = Palette.Accent, unfocusedBorderColor = Palette.Outline),
        )
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            PrimaryButton(if (hi) "पूछें" else "Ask", Modifier.weight(1f), enabled = !ui.explaining && text.isNotBlank()) {
                vm.ask(text.trim())
            }
            val micLabel = when {
                ui.listening -> if (hi) "रुकें" else "Stop"
                ui.transcribing -> if (hi) "सुन रहा है…" else "Listening…"
                else -> if (hi) "बोलें" else "Speak"
            }
            OutlinedButton(
                onClick = {
                    when {
                        ui.listening -> vm.stopVoice()
                        context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED -> vm.startVoice()
                        else -> mic.launch(Manifest.permission.RECORD_AUDIO)
                    }
                },
                enabled = !ui.transcribing,
                modifier = Modifier.weight(1f).height(48.dp),
                shape = MaterialTheme.shapes.small,
                border = androidx.compose.foundation.BorderStroke(1.dp, if (ui.listening) Palette.Warning else Palette.Outline),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = if (ui.listening) Palette.Warning else Palette.Ink),
            ) {
                Icon(AppIcon.Mic, if (ui.listening) Palette.Warning else Palette.Ink, 18.dp)
                Spacer(Modifier.width(6.dp))
                Text(micLabel, style = MaterialTheme.typography.labelLarge)
            }
        }
        ui.transcript?.let {
            Text((if (hi) "सुना: " else "Heard: ") + it, style = MaterialTheme.typography.bodyMedium, color = Palette.InkMuted)
        }
        if (ui.voiceChoices.isNotEmpty()) {
            SectionLabel(if (hi) "आपका मतलब?" else "Did you mean?")
            ui.voiceChoices.forEach { id ->
                Row(
                    Modifier.fillMaxWidth().clip(MaterialTheme.shapes.small)
                        .border(1.dp, Palette.Outline, MaterialTheme.shapes.small)
                        .clickable { vm.chooseVoice(id) }.padding(14.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Box(Modifier.size(10.dp).clip(CircleShape).background(Bins.colour(vm.binColourOf(id))))
                    Spacer(Modifier.width(12.dp))
                    Text(vm.nameOf(id), style = MaterialTheme.typography.bodyLarge)
                }
            }
        }
        if (ui.verdict != null) VerdictCard(vm, Modifier.weight(1f), onWrong = {})
    }
}

// ---------------------------------------------------------------------------------------- impact

private val STREAM_NAMES = mapOf(
    "wet" to ("Wet" to "गीला"), "dry" to ("Dry" to "सूखा"),
    "sanitary" to ("Sanitary" to "सैनिटरी"), "special_care" to ("Special care" to "विशेष देखभाल"),
)

@Suppress("UNCHECKED_CAST")
@Composable
fun ImpactScreen(vm: AppViewModel) {
    val ui = vm.ui
    val hi = ui.language == "hi"
    LazyColumn(Modifier.fillMaxSize().padding(horizontal = 16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item {
            Spacer(Modifier.height(4.dp))
            Text(if (hi) "आपका असर" else "Your impact", style = MaterialTheme.typography.headlineSmall)
        }
        val s = ui.impact
        if (s == null || (s["scans"] as? Number)?.toInt() == 0) {
            item {
                Panel {
                    Text(
                        if (hi) "अभी तक कोई स्कैन नहीं। कोई चीज़ स्कैन करें, यहाँ आपकी छँटाई दिखेगी।"
                        else "No scans yet. Scan an item and your sorting shows up here.",
                        style = MaterialTheme.typography.bodyMedium, color = Palette.InkMuted,
                    )
                }
            }
        } else {
            item {
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    StatTile(if (hi) "छाँटी गई चीज़ें" else "Items sorted", "${s["scans"]}", Modifier.weight(1f))
                    StatTile(if (hi) "लगातार दिन" else "Day streak", "${s["current_streak_days"]}", Modifier.weight(1f))
                }
            }
            item {
                StatTile(
                    if (hi) "लैंडफ़िल से बचाया (अनुमान)" else "Kept out of landfill (estimate)",
                    "~${s["diverted_kg_estimate"]} kg · ${s["diverted_items"]} " + (if (hi) "चीज़ें" else "items"),
                    Modifier.fillMaxWidth(),
                )
            }
            val byStream = (s["by_stream"] as? Map<String, Any?>).orEmpty()
            val total = byStream.values.sumOf { (it as? Number)?.toInt() ?: 0 }.coerceAtLeast(1)
            item {
                Panel {
                    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        SectionLabel(if (hi) "डिब्बे के हिसाब से" else "By bin")
                        listOf("wet", "dry", "sanitary", "special_care").forEach { stream ->
                            val n = (byStream[stream] as? Number)?.toInt() ?: 0
                            val label = STREAM_NAMES[stream]?.let { if (hi) it.second else it.first } ?: stream
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text(label, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.width(96.dp))
                                Box(Modifier.weight(1f).height(10.dp).clip(RoundedCornerShape(50)).background(Palette.SurfaceStrong)) {
                                    Box(
                                        Modifier.fillMaxHeight().fillMaxWidth(n.toFloat() / total)
                                            .clip(RoundedCornerShape(50)).background(Bins.forStream(stream)),
                                    )
                                }
                                Text("$n", style = MaterialTheme.typography.labelLarge, modifier = Modifier.width(32.dp), textAlign = TextAlign.End)
                            }
                        }
                        Text(
                            if (hi) "वज़न ज्ञान-आधार के सामान्य अनुमान से।" else "Weights are typical values from the knowledge base.",
                            style = MaterialTheme.typography.labelSmall, color = Palette.InkMuted,
                        )
                    }
                }
            }
        }
        ui.didYouKnow?.let { d ->
            item {
                Panel {
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        SectionLabel("${d["heading"]}")
                        Text("${d["title"]}", style = MaterialTheme.typography.titleMedium)
                        Text(d["text"] as String, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
        }
        ui.quiz?.let { q -> item { QuizCard(q, hi) } }
        item { Spacer(Modifier.height(8.dp)) }
    }
}

@Composable
private fun StatTile(label: String, value: String, modifier: Modifier) {
    Column(modifier.clip(MaterialTheme.shapes.medium).background(Palette.Surface).padding(16.dp)) {
        Text(value, style = MaterialTheme.typography.titleLarge, color = Palette.Accent)
        Spacer(Modifier.height(2.dp))
        Text(label, style = MaterialTheme.typography.bodyMedium, color = Palette.InkMuted)
    }
}

@Suppress("UNCHECKED_CAST")
@Composable
fun QuizCard(q: Map<String, Any?>, hi: Boolean) {
    var picked by remember(q) { mutableStateOf<String?>(null) }
    Panel {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionLabel(if (hi) "आज का सवाल" else "Today's quiz")
            Text(q["question"] as String, style = MaterialTheme.typography.titleMedium)
            (q["options"] as List<Map<String, Any?>>).forEach { o ->
                val stream = o["stream"] as String
                val isAnswer = stream == q["answer"]
                val border = when {
                    picked == null -> Palette.Outline
                    isAnswer -> Palette.Accent
                    picked == stream -> Palette.Warning
                    else -> Palette.Outline
                }
                Row(
                    Modifier.fillMaxWidth().clip(MaterialTheme.shapes.small).background(Palette.Background)
                        .border(1.dp, border, MaterialTheme.shapes.small)
                        .clickable(enabled = picked == null) { picked = stream }.padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Box(Modifier.size(10.dp).clip(CircleShape).background(Bins.forStream(stream)))
                    Spacer(Modifier.width(10.dp))
                    Text(o["label"] as String, style = MaterialTheme.typography.bodyLarge)
                }
            }
            picked?.let {
                val right = it == q["answer"]
                Text(
                    (if (right) (if (hi) "सही! " else "Correct! ") else (if (hi) "पूरा सही नहीं। " else "Not quite. ")) + q["explanation"],
                    style = MaterialTheme.typography.bodyMedium,
                    color = if (right) Palette.Accent else Palette.Warning,
                )
            }
        }
    }
}

// -------------------------------------------------------------------------------------- settings

@Composable
fun SettingsScreen(vm: AppViewModel) {
    val ui = vm.ui
    val hi = ui.language == "hi"
    var exported by remember { mutableStateOf<String?>(null) }
    Column(
        Modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(if (hi) "सेटिंग" else "Settings", style = MaterialTheme.typography.headlineSmall)

        SectionLabel(if (hi) "भाषा" else "Language")
        Row(
            Modifier.fillMaxWidth().clip(MaterialTheme.shapes.small).background(Palette.Surface).padding(4.dp),
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            listOf("en" to "English", "hi" to "हिंदी").forEach { (code, label) ->
                val selected = ui.language == code
                Box(
                    Modifier.weight(1f).clip(MaterialTheme.shapes.small)
                        .background(if (selected) Palette.Background else Color.Transparent)
                        .clickable { if (!selected) vm.setLanguage(code) }.padding(vertical = 10.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        label, style = MaterialTheme.typography.labelLarge,
                        color = if (selected) Palette.Accent else Palette.InkMuted,
                    )
                }
            }
        }

        SectionLabel(if (hi) "मॉडल (सब इसी फ़ोन पर, बिना इंटरनेट)" else "Models (all on this phone, no internet)")
        Panel(padding = 12.dp) {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                ui.models.forEach { (name, status) ->
                    val ok = status.startsWith("ready")
                    val busy = status.startsWith("loading")
                    Row(verticalAlignment = Alignment.Top) {
                        Box(
                            Modifier.padding(top = 6.dp).size(8.dp).clip(CircleShape)
                                .background(if (ok) Palette.Accent else if (busy) Palette.Amber else Palette.Warning),
                        )
                        Spacer(Modifier.width(10.dp))
                        Column {
                            Text(name, style = MaterialTheme.typography.titleMedium)
                            Text(status, style = MaterialTheme.typography.bodyMedium, color = Palette.InkMuted)
                        }
                    }
                }
            }
        }

        SectionLabel(if (hi) "मापन" else "Measurements")
        Panel(padding = 12.dp) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    if (hi) "समय दिखाएँ" else "Show timings on screen",
                    style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f),
                )
                Switch(
                    checked = ui.showOverlay, onCheckedChange = { vm.toggleOverlay() },
                    colors = SwitchDefaults.colors(checkedTrackColor = Palette.Accent),
                )
            }
        }
        SecondaryButton(if (hi) "रिपोर्ट के लिए समय सहेजें" else "Export timings for the report", Modifier.fillMaxWidth()) {
            exported = vm.exportTimings()
        }
        exported?.let { Text("Saved: $it", style = MaterialTheme.typography.labelSmall, color = Palette.InkMuted) }

        HorizontalDivider(color = Palette.Outline)
        Text(
            "Built with Llama. Llama 3.2 is licensed under the Llama 3.2 Community License, " +
                "Copyright © Meta Platforms, Inc. All Rights Reserved.",
            style = MaterialTheme.typography.labelSmall, color = Palette.InkMuted,
        )
    }
}
