package org.wastesegregation.app

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** Calm palette: white background, a muted sage accent, soft greys. */
object Palette {
    val Background = Color(0xFFFFFFFF)
    val Surface = Color(0xFFF5F6F4)      // cards and panels
    val SurfaceStrong = Color(0xFFECEEEA) // pressed / selected
    val Outline = Color(0xFFDCE0DA)
    val Ink = Color(0xFF1E2622)          // main text
    val InkMuted = Color(0xFF66706A)     // secondary text
    val Accent = Color(0xFF3F6B5B)       // sage green: primary actions
    val AccentSoft = Color(0xFFE3ECE7)
    val Warning = Color(0xFF9A5B3F)      // muted terracotta: hazards
    val WarningSoft = Color(0xFFF6ECE6)
    val Amber = Color(0xFFB08A3E)        // "looking closer" state
}

/**
 * SWM Rules 2026 bin colours, toned down but still unmistakable: green wet, blue dry,
 * red sanitary, black special care. Keys are the KB's bin colour ids.
 */
object Bins {
    private val colours = mapOf(
        "green" to Color(0xFF4E7D5B),
        "blue" to Color(0xFF3D6890),
        "red" to Color(0xFFA2514C),
        "black" to Color(0xFF34393B),
    )
    private val streamToBin = mapOf("wet" to "green", "dry" to "blue", "sanitary" to "red", "special_care" to "black")

    fun colour(bin: String?): Color = colours[bin] ?: Palette.InkMuted
    fun forStream(stream: String): Color = colour(streamToBin[stream])
}

private val AppColours = lightColorScheme(
    primary = Palette.Accent,
    onPrimary = Color.White,
    primaryContainer = Palette.AccentSoft,
    onPrimaryContainer = Palette.Ink,
    secondary = Palette.InkMuted,
    onSecondary = Color.White,
    background = Palette.Background,
    onBackground = Palette.Ink,
    surface = Palette.Background,
    onSurface = Palette.Ink,
    surfaceVariant = Palette.Surface,
    onSurfaceVariant = Palette.InkMuted,
    surfaceContainer = Palette.Surface,
    surfaceContainerLow = Palette.Background,
    surfaceContainerHigh = Palette.Surface,
    outline = Palette.Outline,
    outlineVariant = Palette.Outline,
    error = Palette.Warning,
)

private val AppType = Typography(
    headlineSmall = TextStyle(fontSize = 24.sp, fontWeight = FontWeight.SemiBold, lineHeight = 30.sp),
    titleLarge = TextStyle(fontSize = 20.sp, fontWeight = FontWeight.SemiBold, lineHeight = 26.sp),
    titleMedium = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.SemiBold, lineHeight = 22.sp),
    bodyLarge = TextStyle(fontSize = 16.sp, lineHeight = 23.sp),
    bodyMedium = TextStyle(fontSize = 14.sp, lineHeight = 20.sp),
    labelLarge = TextStyle(fontSize = 14.sp, fontWeight = FontWeight.Medium),
    labelMedium = TextStyle(fontSize = 12.sp, fontWeight = FontWeight.Medium, letterSpacing = 0.3.sp),
    labelSmall = TextStyle(fontSize = 11.sp, letterSpacing = 0.3.sp),
)

private val AppShapes = Shapes(
    small = RoundedCornerShape(10.dp),
    medium = RoundedCornerShape(14.dp),
    large = RoundedCornerShape(20.dp),
)

@Composable
fun AppTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = AppColours, typography = AppType, shapes = AppShapes, content = content)
}
