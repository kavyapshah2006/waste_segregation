package org.wastesegregation.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
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
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executor

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        setContent { AppTheme { App() } }
    }
}

private data class Tab(val en: String, val hi: String, val icon: AppIcon)

private val TABS = listOf(
    Tab("Scan", "स्कैन", AppIcon.Scan),
    Tab("Ask", "पूछें", AppIcon.Ask),
    Tab("History", "इतिहास", AppIcon.History),
    Tab("Impact", "असर", AppIcon.Impact),
)

@Composable
fun App(vm: AppViewModel = viewModel()) {
    var tab by remember { mutableStateOf(0) }
    var settingsOpen by remember { mutableStateOf(false) }
    val ui = vm.ui
    val hi = ui.language == "hi"
    BackHandler(enabled = settingsOpen || (tab == 0 && ui.verdict != null) || tab != 0) {
        when {
            settingsOpen -> settingsOpen = false
            tab == 0 && ui.verdict != null -> vm.scanAgain()
            else -> tab = 0
        }
    }
    Scaffold(
        containerColor = Palette.Page,
        contentWindowInsets = WindowInsets(0, 0, 0, 0),
        bottomBar = { if (!settingsOpen) BottomBar(tab, hi) { tab = it } },
    ) { padding ->
        Box(Modifier.padding(padding).fillMaxSize()) {
            when {
                settingsOpen -> SettingsScreen(vm, onClose = { settingsOpen = false })
                tab == 0 && ui.verdict != null -> ResultScreen(vm)
                tab == 0 -> ScanScreen(vm, onSettings = { settingsOpen = true })
                tab == 1 -> AskScreen(vm)
                tab == 2 -> HistoryScreen(vm)
                else -> ImpactScreen(vm)
            }
        }
    }
    ui.message?.let { msg ->
        AlertDialog(
            onDismissRequest = vm::clearMessage,
            containerColor = Palette.Card,
            confirmButton = { TextButton(onClick = vm::clearMessage) { Text("OK", color = Palette.Teal500) } },
            text = { Text(msg, color = Palette.Ink) },
        )
    }
}

@Composable
private fun BottomBar(tab: Int, hi: Boolean, onTab: (Int) -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(topStart = 22.dp, topEnd = 22.dp))
            .background(Palette.Teal900)
            .navigationBarsPadding()
            .padding(top = 10.dp, bottom = 8.dp),
        horizontalArrangement = Arrangement.SpaceAround,
    ) {
        TABS.forEachIndexed { i, t ->
            val selected = tab == i
            val colour = if (selected) Palette.Mint else Color(0xFFB5C4BF)
            Column(
                Modifier.clip(MaterialTheme.shapes.small).clickable { onTab(i) }.padding(horizontal = 14.dp, vertical = 4.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Icon(t.icon, colour, 24.dp)
                Spacer(Modifier.height(3.dp))
                Text(if (hi) t.hi else t.en, style = MaterialTheme.typography.labelMedium, color = colour)
            }
        }
    }
}

// ---------------------------------------------------------------------------------------- shared

/** Deep teal header with the status bar inset, a title row and optional content below. */
@Composable
private fun Header(
    title: String,
    onBack: (() -> Unit)? = null,
    action: (@Composable () -> Unit)? = null,
    content: (@Composable ColumnScope.() -> Unit)? = null,
) {
    Column(
        Modifier
            .fillMaxWidth()
            .background(Palette.HeaderGradient)
            .statusBarsPadding()
            .padding(start = 16.dp, end = 16.dp, top = 8.dp, bottom = if (content == null) 18.dp else 30.dp),
    ) {
        Row(Modifier.fillMaxWidth().height(44.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(40.dp), contentAlignment = Alignment.Center) {
                if (onBack != null) {
                    Box(Modifier.clip(CircleShape).clickable(onClick = onBack).padding(6.dp)) { Icon(AppIcon.Back, Color.White, 24.dp) }
                }
            }
            Text(
                title, style = MaterialTheme.typography.titleLarge, color = Color.White,
                textAlign = TextAlign.Center, modifier = Modifier.weight(1f),
            )
            Box(Modifier.size(40.dp), contentAlignment = Alignment.Center) { action?.invoke() }
        }
        if (content != null) {
            Spacer(Modifier.height(12.dp))
            content()
        }
    }
}

/** White sheet that slightly overlaps the header, like the reference result screen. */
@Composable
private fun Sheet(content: @Composable ColumnScope.() -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .offset(y = (-18).dp)
            .clip(RoundedCornerShape(topStart = 26.dp, topEnd = 26.dp))
            .background(Palette.Page)
            .padding(horizontal = 16.dp, vertical = 18.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
        content = content,
    )
}

@Composable
private fun WhiteCard(modifier: Modifier = Modifier, content: @Composable ColumnScope.() -> Unit) {
    Column(
        modifier
            .fillMaxWidth()
            .clip(MaterialTheme.shapes.medium)
            .background(Palette.Card)
            .border(1.dp, Palette.Outline, MaterialTheme.shapes.medium)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        content = content,
    )
}

@Composable
fun PrimaryButton(text: String, modifier: Modifier = Modifier, enabled: Boolean = true, onClick: () -> Unit) {
    Button(
        onClick = onClick, enabled = enabled, modifier = modifier.height(52.dp), shape = MaterialTheme.shapes.small,
        colors = ButtonDefaults.buttonColors(containerColor = Palette.Teal800, contentColor = Color.White),
    ) { Text(text, style = MaterialTheme.typography.labelLarge) }
}

@Composable
fun SecondaryButton(text: String, modifier: Modifier = Modifier, enabled: Boolean = true, onClick: () -> Unit) {
    OutlinedButton(
        onClick = onClick, enabled = enabled, modifier = modifier.height(52.dp), shape = MaterialTheme.shapes.small,
        border = BorderStroke(1.dp, Palette.Outline),
        colors = ButtonDefaults.outlinedButtonColors(containerColor = Palette.Card, contentColor = Palette.Ink),
    ) { Text(text, style = MaterialTheme.typography.labelLarge) }
}

@Composable
private fun Chip(text: String, bg: Color, fg: Color, leading: (@Composable () -> Unit)? = null) {
    Row(
        Modifier.clip(RoundedCornerShape(50)).background(bg).padding(horizontal = 12.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (leading != null) {
            leading()
            Spacer(Modifier.width(6.dp))
        }
        Text(text, style = MaterialTheme.typography.labelLarge, color = fg)
    }
}

@Composable
private fun RoundIconButton(icon: AppIcon, size: Dp, bg: Color, tint: Color, border: Color? = null, onClick: () -> Unit) {
    Box(
        Modifier
            .size(size)
            .clip(CircleShape)
            .background(bg)
            .then(if (border != null) Modifier.border(1.5.dp, border, CircleShape) else Modifier)
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center,
    ) { Icon(icon, tint, size * 0.42f) }
}

// ------------------------------------------------------------------------------------------ scan

@Composable
fun ScanScreen(vm: AppViewModel, onSettings: () -> Unit) {
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
    var picking by remember { mutableStateOf(false) }
    val choosing = ui.suggestions.isNotEmpty() || ui.lookingCloser

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        if (granted) CameraPreview(vm) else {
            Column(Modifier.align(Alignment.Center).padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    if (hi) "स्कैन करने के लिए कैमरा चाहिए" else "The camera is needed to scan",
                    color = Color.White, style = MaterialTheme.typography.bodyLarge,
                )
                Spacer(Modifier.height(14.dp))
                PrimaryButton(if (hi) "कैमरा चालू करें" else "Allow camera") { permission.launch(Manifest.permission.CAMERA) }
            }
        }
        // top: brand + settings, then the hint pill
        Column(
            Modifier.fillMaxWidth().statusBarsPadding().padding(horizontal = 18.dp, vertical = 10.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Image(
                    painterResource(R.drawable.app_logo), "SwachhSathi",
                    Modifier.size(40.dp).clip(RoundedCornerShape(10.dp)),
                )
                Spacer(Modifier.width(10.dp))
                Text(
                    "SwachhSathi", style = MaterialTheme.typography.headlineSmall, color = Color.White,
                    modifier = Modifier.weight(1f),
                )
                Row(
                    Modifier.clip(RoundedCornerShape(50)).background(Color.Black.copy(alpha = 0.35f))
                        .border(1.dp, Color.White.copy(alpha = 0.25f), RoundedCornerShape(50))
                        .clickable(onClick = onSettings).padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(AppIcon.Settings, Color.White, 18.dp)
                    Spacer(Modifier.width(6.dp))
                    Text(if (hi) "सेटिंग" else "Settings", style = MaterialTheme.typography.labelLarge, color = Color.White)
                }
            }
            Spacer(Modifier.height(22.dp))
            HintPill(ui, hi)
        }
        if (granted && !choosing) Viewfinder(Modifier.align(Alignment.Center))

        // bottom: suggestions sheet or the camera controls
        Column(Modifier.align(Alignment.BottomCenter).fillMaxWidth()) {
            if (choosing) {
                SuggestionSheet(vm, hi, onOther = { picking = true })
            } else {
                if (ui.canSuggest) {
                    Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                        Text(
                            if (hi) "पक्का नहीं? सुझाव देखें" else "Not sure? See suggestions",
                            style = MaterialTheme.typography.labelLarge, color = Color.White,
                            modifier = Modifier.clip(RoundedCornerShape(50)).background(Palette.Teal800.copy(alpha = 0.9f))
                                .clickable { vm.suggest() }.padding(horizontal = 16.dp, vertical = 10.dp),
                        )
                    }
                    Spacer(Modifier.height(16.dp))
                }
                Row(
                    Modifier.fillMaxWidth().padding(bottom = 22.dp),
                    horizontalArrangement = Arrangement.SpaceEvenly,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    RoundIconButton(AppIcon.Photo, 56.dp, Color.Black.copy(alpha = 0.35f), Color.White, Color.White.copy(alpha = 0.6f)) {
                        picker.launch("image/*")
                    }
                    Shutter { vm.captureNow() }
                    RoundIconButton(
                        AppIcon.Flash, 56.dp,
                        if (ui.torchOn) Palette.Amber else Color.Black.copy(alpha = 0.35f),
                        if (ui.torchOn) Palette.Teal900 else Color.White, Color.White.copy(alpha = 0.6f),
                    ) { vm.toggleTorch() }
                }
            }
        }
    }
    if (picking) {
        ItemPicker(vm, onPick = { picking = false; vm.choose(it) }, onDismiss = { picking = false })
    }
}

@Composable
private fun HintPill(ui: UiState, hi: Boolean) {
    val text = when {
        ui.lookingCloser -> if (hi) "ध्यान से देख रहे हैं…" else "Looking closer…"
        ui.scanState == "stable" -> ui.liveItem ?: ""
        ui.scanState == "unsure" -> if (hi) "पक्का नहीं… स्थिर रखें" else "Not sure yet… hold steady"
        ui.status.startsWith("Point") -> if (hi) "कैमरा किसी कचरे की चीज़ पर करें" else "Point your camera at a waste item"
        else -> ui.status
    }
    if (text.isEmpty()) return
    Row(
        Modifier.clip(RoundedCornerShape(50)).background(Color.Black.copy(alpha = 0.45f))
            .border(1.dp, Color.White.copy(alpha = 0.2f), RoundedCornerShape(50))
            .padding(horizontal = 18.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (ui.lookingCloser) {
            CircularProgressIndicator(Modifier.size(14.dp), color = Palette.Mint, strokeWidth = 2.dp)
            Spacer(Modifier.width(8.dp))
        }
        Text(text, style = MaterialTheme.typography.bodyLarge, color = Color.White)
    }
    if (ui.showOverlay && ui.detectMs != null) {
        Spacer(Modifier.height(6.dp))
        Text(
            "detector ${"%.0f".format(ui.detectMs)} ms", style = MaterialTheme.typography.labelSmall,
            color = Color.White.copy(alpha = 0.75f),
        )
    }
}

@Composable
private fun Shutter(onClick: () -> Unit) {
    Box(
        Modifier.size(82.dp).clip(CircleShape).border(4.dp, Color.White, CircleShape).clickable(onClick = onClick),
        contentAlignment = Alignment.Center,
    ) { Box(Modifier.size(64.dp).clip(CircleShape).background(Color.White)) }
}

@Composable
private fun Viewfinder(modifier: Modifier) {
    Canvas(modifier.size(260.dp)) {
        val c = Color.White
        val w = size.width
        val l = w * 0.17f
        val sw = 5.dp.toPx()
        for ((x, y, dx, dy) in listOf(
            listOf(0f, 0f, 1f, 1f), listOf(w, 0f, -1f, 1f), listOf(0f, w, 1f, -1f), listOf(w, w, -1f, -1f),
        )) {
            drawLine(c, Offset(x, y), Offset(x + l * dx, y), sw, StrokeCap.Round)
            drawLine(c, Offset(x, y), Offset(x, y + l * dy), sw, StrokeCap.Round)
        }
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
                vm.camera = provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            }, mainExecutor)
        }
    })
}

@Composable
private fun SuggestionSheet(vm: AppViewModel, hi: Boolean, onOther: () -> Unit) {
    val ui = vm.ui
    Column(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(topStart = 26.dp, topEnd = 26.dp))
            .background(Palette.Card)
            .padding(horizontal = 16.dp, vertical = 14.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Box(Modifier.align(Alignment.CenterHorizontally).size(width = 40.dp, height = 4.dp).clip(RoundedCornerShape(50)).background(Palette.Outline))
        Text(if (hi) "क्या यह इनमें से कोई है?" else "Is it one of these?", style = MaterialTheme.typography.titleLarge)
        if (ui.lookingCloser && ui.suggestions.isEmpty()) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 8.dp)) {
                CircularProgressIndicator(Modifier.size(18.dp), color = Palette.Teal500, strokeWidth = 2.dp)
                Spacer(Modifier.width(12.dp))
                Text(
                    if (hi) "मिलती-जुलती चीज़ें ढूँढ रहे हैं…" else "Finding the closest matches…",
                    style = MaterialTheme.typography.bodyLarge, color = Palette.InkMuted,
                )
            }
        }
        ui.suggestions.forEach { s ->
            val bin = vm.binColourOf(s.itemId)
            Row(
                Modifier.fillMaxWidth().clip(MaterialTheme.shapes.small).background(Palette.Page)
                    .clickable { vm.choose(s.itemId) }.padding(horizontal = 14.dp, vertical = 14.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Box(Modifier.size(28.dp).clip(CircleShape).background(Bins.tint(bin)), contentAlignment = Alignment.Center) {
                    Box(Modifier.size(12.dp).clip(CircleShape).background(Bins.colour(bin)))
                }
                Spacer(Modifier.width(12.dp))
                Text(vm.nameOf(s.itemId), style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
                Text("›", style = MaterialTheme.typography.titleLarge, color = Palette.InkMuted)
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            SecondaryButton(if (hi) "कोई और" else "Something else", Modifier.weight(1f), onClick = onOther)
            PrimaryButton(if (hi) "फिर से स्कैन" else "Scan again", Modifier.weight(1f), onClick = vm::scanAgain)
        }
    }
}

// ---------------------------------------------------------------------------------------- result

@Composable
fun ResultScreen(vm: AppViewModel) {
    val context = LocalContext.current
    val ui = vm.ui
    val v = ui.verdict ?: return
    val hi = v.language == "hi"
    var correcting by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        Header(
            title = if (hi) "परिणाम" else "Classification result",
            onBack = vm::scanAgain,
            action = {
                Box(
                    Modifier.clip(CircleShape).clickable {
                        vm.shareText()?.let { text ->
                            val send = Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_TEXT, text)
                            context.startActivity(Intent.createChooser(send, null))
                        }
                    }.padding(6.dp),
                ) { Icon(AppIcon.Share, Color.White, 24.dp) }
            },
        ) { SummaryCard(ui, hi) }
        Sheet { ResultBody(vm, onWrong = { correcting = true }) }
    }
    if (correcting) {
        ItemPicker(vm, onPick = { correcting = false; vm.correct(it) }, onDismiss = { correcting = false })
    }
}

@Composable
private fun SummaryCard(ui: UiState, hi: Boolean) {
    val v = ui.verdict ?: return
    Row(
        Modifier.fillMaxWidth().clip(MaterialTheme.shapes.medium).background(Color.White.copy(alpha = 0.08f))
            .border(1.dp, Color.White.copy(alpha = 0.14f), MaterialTheme.shapes.medium).padding(14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        val image = ui.verdictImage
        Box(
            Modifier.size(104.dp).clip(MaterialTheme.shapes.small).border(2.dp, Color.White, MaterialTheme.shapes.small)
                .background(Bins.tint(v.binColour)),
            contentAlignment = Alignment.Center,
        ) {
            if (image != null) {
                Image(image.asImageBitmap(), v.name, Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
            } else {
                Icon(AppIcon.LeafFilled, Bins.colour(v.binColour), 44.dp)
            }
        }
        Spacer(Modifier.width(16.dp))
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(if (hi) "यह है" else "It's a", style = MaterialTheme.typography.bodyLarge, color = Color.White.copy(alpha = 0.85f))
            Text(v.name, style = MaterialTheme.typography.headlineMedium, color = Color.White)
            Chip(v.streamName, Color.White.copy(alpha = 0.92f), Palette.Teal800)
            ui.confidence?.let { c ->
                Chip(
                    "${(c * 100).toInt()}% " + (if (hi) "भरोसा" else "confidence"),
                    Palette.Mint.copy(alpha = 0.18f), Color.White,
                ) {
                    Box(Modifier.size(18.dp).clip(CircleShape).background(Palette.Mint), contentAlignment = Alignment.Center) {
                        Icon(AppIcon.Check, Palette.Teal900, 14.dp)
                    }
                }
            }
        }
    }
}

/** Bin banner, condition questions, hazard, steps, and Llama's "why". Shared by Scan and Ask. */
@Composable
private fun ResultBody(vm: AppViewModel, onWrong: (() -> Unit)?) {
    val ui = vm.ui
    val v = ui.verdict ?: return
    val hi = v.language == "hi"
    val binColour = Bins.colour(v.binColour)

    // "Dispose in the Blue bin"
    Row(
        Modifier.fillMaxWidth().clip(MaterialTheme.shapes.medium).background(Bins.tint(v.binColour))
            .padding(start = 18.dp, end = 12.dp, top = 16.dp, bottom = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(if (hi) "इसे डालें" else "Dispose in the", style = MaterialTheme.typography.bodyLarge, color = Palette.Ink)
            Text(
                if (hi) "${v.binColourName} डिब्बे में" else "${v.binColourName} bin",
                style = MaterialTheme.typography.headlineMedium, color = Palette.Ink,
            )
            Text(v.streamName, style = MaterialTheme.typography.bodyMedium, color = binColour, fontWeight = FontWeight.Medium)
        }
        BinIllustration(binColour, 84.dp)
    }

    ui.askAttributes.forEach { attr ->
        val q = when (attr) {
            "broken" -> if (hi) "क्या यह टूटा है?" else "Is it broken?"
            else -> if (hi) "क्या यह गंदा है (खाना/तेल लगा)?" else "Is it dirty (food or oil on it)?"
        }
        WhiteCard {
            Text(q, style = MaterialTheme.typography.titleMedium)
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                SecondaryButton(if (hi) "हाँ" else "Yes", Modifier.weight(1f)) { vm.setAttribute(attr, true) }
                SecondaryButton(if (hi) "नहीं" else "No", Modifier.weight(1f)) { vm.setAttribute(attr, false) }
            }
        }
    }

    v.hazard?.let {
        Box(Modifier.fillMaxWidth().clip(MaterialTheme.shapes.small).background(Palette.WarningSoft).padding(14.dp)) {
            Text(it, style = MaterialTheme.typography.bodyLarge, color = Palette.Warning, fontWeight = FontWeight.Medium)
        }
    }

    if (v.steps.isNotEmpty() || v.dropOffText != null) {
        WhiteCard {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(AppIcon.Clipboard, Palette.Ink, 24.dp)
                Spacer(Modifier.width(10.dp))
                Text(if (hi) "कैसे फेंकें" else "How to dispose", style = MaterialTheme.typography.titleLarge)
            }
            v.steps.forEach { step ->
                Row(verticalAlignment = Alignment.Top) {
                    Box(Modifier.size(28.dp).clip(CircleShape).background(Palette.MintSoft), contentAlignment = Alignment.Center) {
                        Icon(AppIcon.Check, Palette.Teal500, 18.dp)
                    }
                    Spacer(Modifier.width(12.dp))
                    Text(step, style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(top = 3.dp))
                }
            }
            v.dropOffText?.let { Text(it, style = MaterialTheme.typography.bodyMedium, color = Palette.InkMuted) }
            v.notes.forEach { Text(it, style = MaterialTheme.typography.bodyMedium, color = Palette.InkMuted) }
        }
    }

    // "Why it matters": Llama's explanation, checked against the KB
    Column(
        Modifier.fillMaxWidth().clip(MaterialTheme.shapes.medium).background(Palette.MintSoft).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(AppIcon.LeafFilled, Palette.Teal500, 26.dp)
            Spacer(Modifier.width(10.dp))
            Text(if (hi) "यह क्यों ज़रूरी है" else "Why it matters", style = MaterialTheme.typography.titleLarge, color = Palette.Ink)
        }
        ui.question?.let { Text(it, style = MaterialTheme.typography.titleMedium, color = Palette.MintInk) }
        val verified = ui.verified
        when {
            verified != null -> {
                verified.output["why"]?.let { Text(it as String, style = MaterialTheme.typography.bodyLarge, color = Palette.MintInk) }
                verified.output["tip"]?.let {
                    Text((if (hi) "सुझाव: " else "Tip: ") + it, style = MaterialTheme.typography.bodyMedium, color = Palette.MintInk)
                }
                if (verified.status != "ok") {
                    Text(
                        if (hi) "ज्ञान-आधार से दिखाया गया" else "Shown from the knowledge base",
                        style = MaterialTheme.typography.labelSmall, color = Palette.InkMuted,
                    )
                }
            }
            ui.explaining -> Row(verticalAlignment = Alignment.Top) {
                CircularProgressIndicator(Modifier.padding(top = 3.dp).size(14.dp), color = Palette.Teal500, strokeWidth = 2.dp)
                Spacer(Modifier.width(10.dp))
                Text(
                    ui.whyStream.ifEmpty { if (hi) "Llama इसी फ़ोन पर समझा रहा है…" else "Llama is explaining, on this phone…" },
                    style = MaterialTheme.typography.bodyLarge, color = Palette.MintInk,
                    fontStyle = if (ui.whyStream.isEmpty()) FontStyle.Italic else FontStyle.Normal,
                )
            }
            else -> Text("–", color = Palette.InkMuted)
        }
    }

    if (ui.showOverlay) {
        Text(
            "card ${ui.cardMs?.let { "%.0f ms".format(it) } ?: "–"} · Llama ${ui.firstTokenS?.let { "%.1f s".format(it) } ?: "–"} first token, " +
                "${ui.decodeTps?.let { "%.1f".format(it) } ?: "–"} tok/s",
            style = MaterialTheme.typography.labelSmall, color = Palette.InkMuted,
            textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth(),
        )
    }
    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        if (onWrong != null) SecondaryButton(if (hi) "यह गलत है" else "This is wrong", Modifier.weight(1f), onClick = onWrong)
        PrimaryButton(if (hi) "फिर से स्कैन" else "Scan again", Modifier.weight(1f), onClick = vm::scanAgain)
    }
    Spacer(Modifier.height(8.dp))
}

@Composable
fun ItemPicker(vm: AppViewModel, onPick: (String) -> Unit, onDismiss: () -> Unit) {
    var filter by remember { mutableStateOf("") }
    val all = remember { vm.itemChoices() }
    val hi = vm.ui.language == "hi"
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = Palette.Card,
        confirmButton = { TextButton(onClick = onDismiss) { Text(if (hi) "रद्द करें" else "Cancel", color = Palette.Teal500) } },
        title = { Text(if (hi) "यह क्या है?" else "What is it?", style = MaterialTheme.typography.titleLarge) },
        text = {
            Column {
                OutlinedTextField(
                    filter, { filter = it }, label = { Text(if (hi) "खोजें" else "Search") }, singleLine = true,
                    shape = MaterialTheme.shapes.small, modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = Palette.Teal500, unfocusedBorderColor = Palette.Outline),
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
    val mic = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok -> if (ok) vm.startVoice() }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        Header(if (hi) "पूछें" else "Ask") {
            Text(
                ui.verdict?.let { (if (hi) "अभी: " else "About: ") + it.name }
                    ?: if (hi) "कोई चीज़ स्कैन करें, या सवाल में उसका नाम लिखें या बोलें।"
                    else "Scan an item, or name it in your question. Type or speak, English or Hindi.",
                style = MaterialTheme.typography.bodyLarge, color = Color.White.copy(alpha = 0.85f),
            )
        }
        Sheet {
            WhiteCard {
                OutlinedTextField(
                    text, { text = it }, Modifier.fillMaxWidth(),
                    label = { Text(if (hi) "आपका सवाल" else "Your question") },
                    placeholder = { Text(if (hi) "पुरानी बैटरी कहाँ डालूँ?" else "Where does an old battery go?") },
                    shape = MaterialTheme.shapes.small,
                    colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = Palette.Teal500, unfocusedBorderColor = Palette.Outline),
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
                        modifier = Modifier.weight(1f).height(52.dp), shape = MaterialTheme.shapes.small,
                        border = BorderStroke(1.dp, if (ui.listening) Palette.Warning else Palette.Outline),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = if (ui.listening) Palette.Warning else Palette.Ink),
                    ) {
                        Icon(AppIcon.Mic, if (ui.listening) Palette.Warning else Palette.Ink, 18.dp)
                        Spacer(Modifier.width(6.dp))
                        Text(micLabel, style = MaterialTheme.typography.labelLarge)
                    }
                }
                ui.transcript?.let { Text((if (hi) "सुना: " else "Heard: ") + it, style = MaterialTheme.typography.bodyMedium, color = Palette.InkMuted) }
            }
            if (ui.voiceChoices.isNotEmpty()) {
                Text(if (hi) "आपका मतलब?" else "Did you mean?", style = MaterialTheme.typography.titleMedium)
                ui.voiceChoices.forEach { id ->
                    Row(
                        Modifier.fillMaxWidth().clip(MaterialTheme.shapes.small).background(Palette.Card)
                            .clickable { vm.chooseVoice(id) }.padding(14.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Box(Modifier.size(12.dp).clip(CircleShape).background(Bins.colour(vm.binColourOf(id))))
                        Spacer(Modifier.width(12.dp))
                        Text(vm.nameOf(id), style = MaterialTheme.typography.bodyLarge)
                    }
                }
            }
            if (ui.verdict != null) ResultBody(vm, onWrong = null)
        }
    }
}

// --------------------------------------------------------------------------------------- history

@Composable
fun HistoryScreen(vm: AppViewModel) {
    val ui = vm.ui
    val hi = ui.language == "hi"
    Column(Modifier.fillMaxSize()) {
        Header(if (hi) "इतिहास" else "History")
        if (ui.history.isEmpty()) {
            Sheet {
                WhiteCard {
                    Text(if (hi) "अभी कोई स्कैन नहीं" else "Your scans show up here", style = MaterialTheme.typography.titleMedium)
                    Text(
                        if (hi) "किसी चीज़ को स्कैन करें; उसका डिब्बा यहाँ सहेजा जाएगा, सिर्फ़ इसी फ़ोन पर।"
                        else "Scan an item and its bin is saved here, on this phone only.",
                        style = MaterialTheme.typography.bodyMedium, color = Palette.InkMuted,
                    )
                }
            }
        } else {
            LazyColumn(
                Modifier.fillMaxSize().padding(horizontal = 16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                item { Spacer(Modifier.height(6.dp)) }
                items(ui.history) { row ->
                    Row(
                        Modifier.fillMaxWidth().clip(MaterialTheme.shapes.medium).background(Palette.Card)
                            .border(1.dp, Palette.Outline, MaterialTheme.shapes.medium).padding(14.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Box(Modifier.size(44.dp).clip(CircleShape).background(Bins.tint(row.binColour)), contentAlignment = Alignment.Center) {
                            Box(Modifier.size(16.dp).clip(CircleShape).background(Bins.colour(row.binColour)))
                        }
                        Spacer(Modifier.width(14.dp))
                        Column(Modifier.weight(1f)) {
                            Text(row.name, style = MaterialTheme.typography.titleMedium)
                            Text(
                                if (hi) "${row.binColourName} डिब्बा · ${row.streamName}" else "${row.binColourName} bin · ${row.streamName}",
                                style = MaterialTheme.typography.bodyMedium, color = Palette.InkMuted,
                            )
                        }
                        Text(whenText(row.timeMs, hi), style = MaterialTheme.typography.labelMedium, color = Palette.InkMuted)
                    }
                }
                item { Spacer(Modifier.height(10.dp)) }
            }
        }
    }
}

private fun whenText(timeMs: Long, hi: Boolean): String {
    val then = Calendar.getInstance().apply { timeInMillis = timeMs }
    val now = Calendar.getInstance()
    val sameDay = then.get(Calendar.YEAR) == now.get(Calendar.YEAR) && then.get(Calendar.DAY_OF_YEAR) == now.get(Calendar.DAY_OF_YEAR)
    val time = SimpleDateFormat("HH:mm", Locale.US).format(Date(timeMs))
    return if (sameDay) (if (hi) "आज $time" else "Today $time") else SimpleDateFormat("d MMM", Locale.US).format(Date(timeMs))
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
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        val s = ui.impact
        Header(if (hi) "आपका असर" else "Your impact") {
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                HeaderStat("${s?.get("scans") ?: 0}", if (hi) "चीज़ें छाँटीं" else "Items sorted", Modifier.weight(1f))
                HeaderStat("~${s?.get("diverted_kg_estimate") ?: 0} kg", if (hi) "लैंडफ़िल से बचाया" else "Kept from landfill", Modifier.weight(1f))
                HeaderStat("${s?.get("current_streak_days") ?: 0}", if (hi) "लगातार दिन" else "Day streak", Modifier.weight(1f))
            }
        }
        Sheet {
            val byStream = (s?.get("by_stream") as? Map<String, Any?>).orEmpty()
            val total = byStream.values.sumOf { (it as? Number)?.toInt() ?: 0 }.coerceAtLeast(1)
            WhiteCard {
                Text(if (hi) "डिब्बे के हिसाब से" else "By bin", style = MaterialTheme.typography.titleLarge)
                listOf("wet", "dry", "sanitary", "special_care").forEach { stream ->
                    val n = (byStream[stream] as? Number)?.toInt() ?: 0
                    val label = STREAM_NAMES[stream]?.let { if (hi) it.second else it.first } ?: stream
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(label, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.width(100.dp))
                        Box(Modifier.weight(1f).height(10.dp).clip(RoundedCornerShape(50)).background(Palette.Page)) {
                            Box(Modifier.fillMaxHeight().fillMaxWidth(n.toFloat() / total).clip(RoundedCornerShape(50)).background(Bins.forStream(stream)))
                        }
                        Text("$n", style = MaterialTheme.typography.labelLarge, modifier = Modifier.width(34.dp), textAlign = TextAlign.End)
                    }
                }
                Text(
                    if (hi) "वज़न ज्ञान-आधार के सामान्य अनुमान से।" else "Weights are typical values from the knowledge base.",
                    style = MaterialTheme.typography.labelSmall, color = Palette.InkMuted,
                )
            }
            Text(if (hi) "सीखें" else "Learn", style = MaterialTheme.typography.titleLarge)
            ui.didYouKnow?.let { d ->
                Column(
                    Modifier.fillMaxWidth().clip(MaterialTheme.shapes.medium).background(Palette.MintSoft).padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    Text("${d["heading"]}", style = MaterialTheme.typography.labelMedium, color = Palette.Teal500)
                    Text("${d["title"]}", style = MaterialTheme.typography.titleMedium, color = Palette.Ink)
                    Text(d["text"] as String, style = MaterialTheme.typography.bodyMedium, color = Palette.MintInk)
                }
            }
            ui.quiz?.let { q -> QuizCard(q, hi) }
        }
    }
}

@Composable
private fun HeaderStat(value: String, label: String, modifier: Modifier) {
    Column(
        modifier.clip(MaterialTheme.shapes.small).background(Color.White.copy(alpha = 0.08f))
            .border(1.dp, Color.White.copy(alpha = 0.14f), MaterialTheme.shapes.small).padding(12.dp),
    ) {
        Text(value, style = MaterialTheme.typography.titleLarge, color = Color.White)
        Text(label, style = MaterialTheme.typography.labelMedium, color = Color.White.copy(alpha = 0.8f))
    }
}

@Suppress("UNCHECKED_CAST")
@Composable
fun QuizCard(q: Map<String, Any?>, hi: Boolean) {
    var picked by remember(q) { mutableStateOf<String?>(null) }
    WhiteCard {
        Text(if (hi) "आज का सवाल" else "Today's quiz", style = MaterialTheme.typography.labelMedium, color = Palette.Teal500)
        Text(q["question"] as String, style = MaterialTheme.typography.titleMedium)
        (q["options"] as List<Map<String, Any?>>).forEach { o ->
            val stream = o["stream"] as String
            val isAnswer = stream == q["answer"]
            val border = when {
                picked == null -> Palette.Outline
                isAnswer -> Palette.Teal500
                picked == stream -> Palette.Warning
                else -> Palette.Outline
            }
            Row(
                Modifier.fillMaxWidth().clip(MaterialTheme.shapes.small).border(1.5.dp, border, MaterialTheme.shapes.small)
                    .clickable(enabled = picked == null) { picked = stream }.padding(12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Box(Modifier.size(12.dp).clip(CircleShape).background(Bins.forStream(stream)))
                Spacer(Modifier.width(10.dp))
                Text(o["label"] as String, style = MaterialTheme.typography.bodyLarge)
            }
        }
        picked?.let {
            val right = it == q["answer"]
            Text(
                (if (right) (if (hi) "सही! " else "Correct! ") else (if (hi) "पूरा सही नहीं। " else "Not quite. ")) + q["explanation"],
                style = MaterialTheme.typography.bodyMedium, color = if (right) Palette.Teal500 else Palette.Warning,
            )
        }
    }
}

// -------------------------------------------------------------------------------------- settings

@Composable
fun SettingsScreen(vm: AppViewModel, onClose: () -> Unit) {
    val ui = vm.ui
    val hi = ui.language == "hi"
    var exported by remember { mutableStateOf<String?>(null) }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        Header(if (hi) "सेटिंग" else "Settings", onBack = onClose)
        Sheet {
            WhiteCard {
                Text(if (hi) "भाषा" else "Language", style = MaterialTheme.typography.titleMedium)
                Row(
                    Modifier.fillMaxWidth().clip(MaterialTheme.shapes.small).background(Palette.Page).padding(4.dp),
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    listOf("en" to "English", "hi" to "हिंदी").forEach { (code, label) ->
                        val selected = ui.language == code
                        Box(
                            Modifier.weight(1f).clip(MaterialTheme.shapes.small)
                                .background(if (selected) Palette.Teal800 else Color.Transparent)
                                .clickable { if (!selected) vm.setLanguage(code) }.padding(vertical = 11.dp),
                            contentAlignment = Alignment.Center,
                        ) {
                            Text(label, style = MaterialTheme.typography.labelLarge, color = if (selected) Color.White else Palette.InkMuted)
                        }
                    }
                }
            }
            WhiteCard {
                Text(if (hi) "मॉडल (सब इसी फ़ोन पर, बिना इंटरनेट)" else "Models (all on this phone, no internet)", style = MaterialTheme.typography.titleMedium)
                ui.models.forEach { (name, status) ->
                    val ok = status.startsWith("ready")
                    val busy = status.startsWith("loading")
                    Row(verticalAlignment = Alignment.Top) {
                        Box(
                            Modifier.padding(top = 6.dp).size(9.dp).clip(CircleShape)
                                .background(if (ok) Palette.Teal500 else if (busy) Palette.Amber else Palette.Warning),
                        )
                        Spacer(Modifier.width(10.dp))
                        Column {
                            Text(name, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium)
                            Text(status, style = MaterialTheme.typography.bodyMedium, color = Palette.InkMuted)
                        }
                    }
                }
            }
            WhiteCard {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(if (hi) "समय दिखाएँ" else "Show timings on screen", style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
                    Switch(
                        checked = ui.showOverlay, onCheckedChange = { vm.toggleOverlay() },
                        colors = SwitchDefaults.colors(checkedTrackColor = Palette.Teal500),
                    )
                }
                SecondaryButton(if (hi) "रिपोर्ट के लिए समय सहेजें" else "Export timings for the report", Modifier.fillMaxWidth()) {
                    exported = vm.exportTimings()
                }
                exported?.let { Text("Saved: $it", style = MaterialTheme.typography.labelSmall, color = Palette.InkMuted) }
            }
            Text(
                "Built with Llama. Llama 3.2 is licensed under the Llama 3.2 Community License, " +
                    "Copyright © Meta Platforms, Inc. All Rights Reserved.",
                style = MaterialTheme.typography.labelSmall, color = Palette.InkMuted,
            )
        }
    }
}
