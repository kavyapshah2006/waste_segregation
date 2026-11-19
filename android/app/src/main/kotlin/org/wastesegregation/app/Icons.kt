package org.wastesegregation.app

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/** Simple line icons drawn on a 24-unit grid, so the app needs no icon library. */
enum class AppIcon { Scan, Ask, Impact, Settings, Mic, Photo, Back, Share, Flash, History, Check, Clipboard, LeafFilled }

@Composable
fun Icon(icon: AppIcon, tint: Color, size: Dp = 24.dp) {
    Canvas(Modifier.size(size)) {
        val u = this.size.width / 24f
        val stroke = Stroke(width = 1.8f * u, cap = StrokeCap.Round, join = StrokeJoin.Round)
        when (icon) {
            AppIcon.Scan -> scan(u, tint, stroke)
            AppIcon.Ask -> ask(u, tint, stroke)
            AppIcon.Impact -> impact(u, tint, stroke)
            AppIcon.Settings -> settings(u, tint, stroke)
            AppIcon.Mic -> mic(u, tint, stroke)
            AppIcon.Photo -> photo(u, tint, stroke)
            AppIcon.Back -> { line(u, tint, stroke, 15f, 5f, 8f, 12f); line(u, tint, stroke, 8f, 12f, 15f, 19f) }
            AppIcon.Share -> share(u, tint, stroke)
            AppIcon.Flash -> flash(u, tint, stroke)
            AppIcon.History -> history(u, tint, stroke)
            AppIcon.Check -> { line(u, tint, stroke, 6f, 12.5f, 10f, 16.5f); line(u, tint, stroke, 10f, 16.5f, 18f, 8f) }
            AppIcon.Clipboard -> clipboard(u, tint, stroke)
            AppIcon.LeafFilled -> leafFilled(u, tint)
        }
    }
}

private fun DrawScope.line(u: Float, c: Color, s: Stroke, x1: Float, y1: Float, x2: Float, y2: Float) =
    drawLine(c, Offset(x1 * u, y1 * u), Offset(x2 * u, y2 * u), s.width, StrokeCap.Round)

private fun DrawScope.scan(u: Float, c: Color, s: Stroke) {
    // Four viewfinder corners and a centre dot.
    for ((x, y, dx, dy) in listOf(
        listOf(4f, 4f, 1f, 1f), listOf(20f, 4f, -1f, 1f), listOf(4f, 20f, 1f, -1f), listOf(20f, 20f, -1f, -1f),
    )) {
        line(u, c, s, x, y, x + 5f * dx, y)
        line(u, c, s, x, y, x, y + 5f * dy)
    }
    drawCircle(c, 2.6f * u, Offset(12f * u, 12f * u))
}

private fun DrawScope.ask(u: Float, c: Color, s: Stroke) {
    val p = Path().apply {
        moveTo(5f * u, 5f * u)
        lineTo(19f * u, 5f * u)
        quadraticTo(21f * u, 5f * u, 21f * u, 7f * u)
        lineTo(21f * u, 15f * u)
        quadraticTo(21f * u, 17f * u, 19f * u, 17f * u)
        lineTo(11f * u, 17f * u)
        lineTo(7f * u, 20.5f * u)
        lineTo(7f * u, 17f * u)
        lineTo(5f * u, 17f * u)
        quadraticTo(3f * u, 17f * u, 3f * u, 15f * u)
        lineTo(3f * u, 7f * u)
        quadraticTo(3f * u, 5f * u, 5f * u, 5f * u)
        close()
    }
    drawPath(p, c, style = s)
    line(u, c, s, 8f, 10f, 16f, 10f)
    line(u, c, s, 8f, 13f, 13f, 13f)
}

private fun DrawScope.impact(u: Float, c: Color, s: Stroke) {
    // A leaf with its midrib.
    val p = Path().apply {
        moveTo(5f * u, 19f * u)
        cubicTo(4f * u, 10f * u, 10f * u, 4f * u, 20f * u, 4f * u)
        cubicTo(20f * u, 14f * u, 14f * u, 20f * u, 5f * u, 19f * u)
        close()
    }
    drawPath(p, c, style = s)
    line(u, c, s, 5f, 19f, 15f, 9f)
}

private fun DrawScope.settings(u: Float, c: Color, s: Stroke) {
    // Three sliders.
    for ((y, knob) in listOf(6f to 15f, 12f to 8f, 18f to 13f)) {
        line(u, c, s, 4f, y, 20f, y)
        drawCircle(Color.White, 2.4f * u, Offset(knob * u, y * u))
        drawCircle(c, 2.4f * u, Offset(knob * u, y * u), style = s)
    }
}

private fun DrawScope.mic(u: Float, c: Color, s: Stroke) {
    drawRoundRect(c, Offset(9f * u, 3f * u), Size(6f * u, 11f * u), CornerRadius(3f * u), style = s)
    val p = Path().apply {
        moveTo(6f * u, 11f * u)
        cubicTo(6f * u, 18f * u, 18f * u, 18f * u, 18f * u, 11f * u)
    }
    drawPath(p, c, style = s)
    line(u, c, s, 12f, 17f, 12f, 21f)
}

private fun DrawScope.photo(u: Float, c: Color, s: Stroke) {
    drawRoundRect(c, Offset(3f * u, 5f * u), Size(18f * u, 14f * u), CornerRadius(2.5f * u), style = s)
    val p = Path().apply {
        moveTo(3f * u, 16f * u)
        lineTo(9f * u, 11f * u)
        lineTo(13f * u, 14.5f * u)
        lineTo(16f * u, 12f * u)
        lineTo(21f * u, 16f * u)
    }
    drawPath(p, c, style = s)
    drawCircle(c, 1.5f * u, Offset(16f * u, 9f * u))
}

private fun DrawScope.share(u: Float, c: Color, s: Stroke) {
    val p = Path().apply {
        moveTo(8f * u, 10f * u); lineTo(6f * u, 10f * u); lineTo(6f * u, 20f * u)
        lineTo(18f * u, 20f * u); lineTo(18f * u, 10f * u); lineTo(16f * u, 10f * u)
    }
    drawPath(p, c, style = s)
    line(u, c, s, 12f, 3.5f, 12f, 14f)
    line(u, c, s, 8.5f, 7f, 12f, 3.5f)
    line(u, c, s, 15.5f, 7f, 12f, 3.5f)
}

private fun DrawScope.flash(u: Float, c: Color, s: Stroke) {
    val p = Path().apply {
        moveTo(13f * u, 3f * u); lineTo(6f * u, 13.5f * u); lineTo(11.5f * u, 13.5f * u)
        lineTo(10.5f * u, 21f * u); lineTo(18f * u, 10f * u); lineTo(12.5f * u, 10f * u); close()
    }
    drawPath(p, c, style = s)
}

private fun DrawScope.history(u: Float, c: Color, s: Stroke) {
    drawCircle(c, 8.5f * u, Offset(12f * u, 12f * u), style = s)
    line(u, c, s, 12f, 7.5f, 12f, 12f)
    line(u, c, s, 12f, 12f, 15f, 14f)
}

private fun DrawScope.clipboard(u: Float, c: Color, s: Stroke) {
    drawRoundRect(c, Offset(5f * u, 5f * u), Size(14f * u, 16f * u), CornerRadius(2.5f * u), style = s)
    drawRoundRect(c, Offset(9f * u, 3f * u), Size(6f * u, 4f * u), CornerRadius(1.2f * u), style = s)
}

private fun DrawScope.leafFilled(u: Float, c: Color) {
    val p = Path().apply {
        moveTo(4f * u, 20f * u)
        cubicTo(3f * u, 10f * u, 10f * u, 3.5f * u, 21f * u, 3.5f * u)
        cubicTo(21f * u, 14f * u, 14f * u, 21f * u, 4f * u, 20f * u)
        close()
    }
    drawPath(p, c)
    drawLine(Color.White.copy(alpha = 0.7f), Offset(5f * u, 19f * u), Offset(15f * u, 9f * u), 1.4f * u, StrokeCap.Round)
}

/** A flat wheelie bin in [colour], with a lid and a recycling-style mark, for the bin banner. */
@Composable
fun BinIllustration(colour: Color, size: Dp = 84.dp) {
    Canvas(Modifier.size(size)) {
        val w = this.size.width
        val h = this.size.height
        val dark = Color(
            red = colour.red * 0.78f, green = colour.green * 0.78f, blue = colour.blue * 0.78f, alpha = 1f,
        )
        // body: slightly tapered towards the bottom
        val body = Path().apply {
            moveTo(w * 0.16f, h * 0.30f)
            lineTo(w * 0.84f, h * 0.30f)
            lineTo(w * 0.78f, h * 0.94f)
            quadraticTo(w * 0.77f, h * 0.98f, w * 0.72f, h * 0.98f)
            lineTo(w * 0.28f, h * 0.98f)
            quadraticTo(w * 0.23f, h * 0.98f, w * 0.22f, h * 0.94f)
            close()
        }
        drawPath(body, colour)
        // lid and handle
        drawRoundRect(dark, Offset(w * 0.10f, h * 0.20f), Size(w * 0.80f, h * 0.11f), CornerRadius(w * 0.03f))
        drawRoundRect(dark, Offset(w * 0.38f, h * 0.12f), Size(w * 0.24f, h * 0.09f), CornerRadius(w * 0.03f))
        // three chasing arrows, simplified as chevrons around the centre
        val cx = w * 0.5f
        val cy = h * 0.62f
        val r = w * 0.15f
        val sw = w * 0.045f
        for (k in 0 until 3) {
            val a = Math.toRadians((k * 120 - 90).toDouble())
            val b = Math.toRadians((k * 120 + 10).toDouble())
            val p1 = Offset(cx + r * kotlin.math.cos(a).toFloat(), cy + r * kotlin.math.sin(a).toFloat())
            val p2 = Offset(cx + r * kotlin.math.cos(b).toFloat(), cy + r * kotlin.math.sin(b).toFloat())
            drawLine(Color.White, p1, p2, sw, StrokeCap.Round)
            val tip = Offset(p2.x, p2.y)
            val back = Math.toRadians((k * 120 + 10 - 40).toDouble())
            drawLine(
                Color.White, tip,
                Offset(cx + r * 0.62f * kotlin.math.cos(back).toFloat(), cy + r * 0.62f * kotlin.math.sin(back).toFloat()),
                sw, StrokeCap.Round,
            )
        }
    }
}
