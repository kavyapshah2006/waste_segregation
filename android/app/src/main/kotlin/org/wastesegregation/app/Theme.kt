package org.wastesegregation.app

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** Deep teal (from the marble reference), white cards, mint accents. */
object Palette {
    val Teal900 = Color(0xFF0A2421)   // darkest: bottom bar, camera chrome
    val Teal800 = Color(0xFF0F3B35)   // header
    val Teal700 = Color(0xFF16483F)   // header gradient end
    val Teal500 = Color(0xFF237563)   // buttons, links
    val Mint = Color(0xFF7FD3A6)      // active tab, check marks on dark
    val MintSoft = Color(0xFFDDEFE4)  // "why it matters" panel
    val MintInk = Color(0xFF1F5B47)   // text on mint
    val Page = Color(0xFFEFF2F1)      // result body background
    val Card = Color(0xFFFFFFFF)
    val Outline = Color(0xFFDDE3E0)
    val Ink = Color(0xFF15201D)
    val InkMuted = Color(0xFF66716C)
    val Warning = Color(0xFF9A4A3C)
    val WarningSoft = Color(0xFFF7E9E4)
    val Amber = Color(0xFFD9A441)

    val HeaderGradient = Brush.verticalGradient(listOf(Teal800, Teal700))
}

/**
 * SWM Rules 2026 bin colours, tuned to sit with the teal: green wet (lighter and yellower than
 * the header so a wet card never blends into it), petrol blue dry, red sanitary, charcoal
 * special care. Keys are the KB's bin colour ids.
 */
object Bins {
    private val colours = mapOf(
        "green" to Color(0xFF5B8F45),
        "blue" to Color(0xFF2A5A78),
        "red" to Color(0xFFA84B48),
        "black" to Color(0xFF353B40),
    )
    private val tints = mapOf(
        "green" to Color(0xFFE6F0DF),
        "blue" to Color(0xFFDCE8F1),
        "red" to Color(0xFFF5E2E1),
        "black" to Color(0xFFE5E7E9),
    )
    private val streamToBin = mapOf("wet" to "green", "dry" to "blue", "sanitary" to "red", "special_care" to "black")

    fun colour(bin: String?): Color = colours[bin] ?: Palette.InkMuted
    fun tint(bin: String?): Color = tints[bin] ?: Palette.Page
    fun forStream(stream: String): Color = colour(streamToBin[stream])
}

private val AppColours = lightColorScheme(
    primary = Palette.Teal500,
    onPrimary = Color.White,
    primaryContainer = Palette.MintSoft,
    onPrimaryContainer = Palette.MintInk,
    secondary = Palette.InkMuted,
    onSecondary = Color.White,
    background = Palette.Page,
    onBackground = Palette.Ink,
    surface = Palette.Card,
    onSurface = Palette.Ink,
    surfaceVariant = Palette.Page,
    onSurfaceVariant = Palette.InkMuted,
    surfaceContainer = Palette.Card,
    surfaceContainerLow = Palette.Card,
    surfaceContainerHigh = Palette.Card,
    outline = Palette.Outline,
    outlineVariant = Palette.Outline,
    error = Palette.Warning,
)

private val AppType = Typography(
    headlineMedium = TextStyle(fontSize = 28.sp, fontWeight = FontWeight.Bold, lineHeight = 34.sp),
    headlineSmall = TextStyle(fontSize = 24.sp, fontWeight = FontWeight.Bold, lineHeight = 30.sp),
    titleLarge = TextStyle(fontSize = 20.sp, fontWeight = FontWeight.SemiBold, lineHeight = 26.sp),
    titleMedium = TextStyle(fontSize = 17.sp, fontWeight = FontWeight.SemiBold, lineHeight = 23.sp),
    bodyLarge = TextStyle(fontSize = 16.sp, lineHeight = 23.sp),
    bodyMedium = TextStyle(fontSize = 14.sp, lineHeight = 20.sp),
    labelLarge = TextStyle(fontSize = 15.sp, fontWeight = FontWeight.Medium),
    labelMedium = TextStyle(fontSize = 12.sp, fontWeight = FontWeight.Medium),
    labelSmall = TextStyle(fontSize = 11.sp),
)

private val AppShapes = Shapes(
    small = RoundedCornerShape(12.dp),
    medium = RoundedCornerShape(18.dp),
    large = RoundedCornerShape(26.dp),
)

@Composable
fun AppTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = AppColours, typography = AppType, shapes = AppShapes, content = content)
}
