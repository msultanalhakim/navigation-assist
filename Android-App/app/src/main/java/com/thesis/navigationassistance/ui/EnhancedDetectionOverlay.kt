package com.thesis.navigationassistance.ui

import android.graphics.Paint
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope // Import eksplisit untuk kejelasan
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb // Untuk konversi Color ke Int
import com.thesis.navigationassistance.data.DetectionWithDistance
import com.thesis.navigationassistance.ml.DepthEstimator
import kotlin.math.max
import kotlin.math.sin // Import sin

@Composable
fun EnhancedDetectionOverlayWithDistance(
    detectionsWithDistance: List<DetectionWithDistance>,
    cameraWidth: Int,
    cameraHeight: Int,
    modifier: Modifier = Modifier
) {
    val infiniteTransition = rememberInfiniteTransition(label = "detection_animation")
    // Animasi alpha untuk deteksi paling penting
    val animatedAlpha by infiniteTransition.animateFloat(
        initialValue = 0.6f,
        targetValue = 1.0f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "alpha"
    )

    // Pool untuk mengelola objek Paint agar tidak dibuat ulang terus-menerus
    val paintPool = remember { PaintPool() }

    Canvas(modifier = modifier.fillMaxSize()) {
        val canvasWidth = size.width
        val canvasHeight = size.height

        // Ambil maksimal 5 deteksi teratas berdasarkan confidence
        val topDetections = detectionsWithDistance
            .sortedWith(DetectionWithDistance.ByConfidenceDescending) // Gunakan comparator
            .take(5)

        // Jika tidak ada deteksi, tampilkan indikator pemindaian
        if (topDetections.isEmpty()) {
            drawScanningIndicator(canvasWidth, canvasHeight, paintPool)
            return@Canvas // Keluar dari Canvas jika tidak ada deteksi
        }

        // Hitung skala konversi dari koordinat kamera ke koordinat canvas
        if (cameraWidth <= 0 || cameraHeight <= 0) return@Canvas // Hindari pembagian nol
        val scaleX = canvasWidth / cameraWidth
        val scaleY = canvasHeight / cameraHeight

        topDetections.forEachIndexed { index, detWithDist ->
            val detection = detWithDist.detection
            val distInfo = detWithDist.distanceInfo
            val warningLevel = detWithDist.warningLevel
            val bbox = detection.bbox

            // Lewati jika bbox tidak valid
            if (!bbox.isValid()) return@forEachIndexed

            // Konversi koordinat bbox ke canvas
            val displayLeft = bbox.x1 * scaleX
            val displayTop = bbox.y1 * scaleY
            val displayRight = bbox.x2 * scaleX
            val displayBottom = bbox.y2 * scaleY

            // Lewati jika objek sepenuhnya di luar layar
            if (displayRight < 0 || displayLeft > canvasWidth || displayBottom < 0 || displayTop > canvasHeight) {
                return@forEachIndexed
            }

            // Tentukan warna dasar dan nama tampilan
            val (boxColor, displayName) = getColorAndName(detection.className)
            // Tentukan warna peringatan berdasarkan level bahaya
            val warningColor = getWarningColor(warningLevel, boxColor)

            // Tentukan alpha (lebih terang untuk deteksi utama)
            val alpha = if (index == 0) animatedAlpha else (detection.confidence * 0.5f + 0.5f).coerceIn(0.6f, 0.9f)
            val colorWithAlpha = warningColor.copy(alpha = alpha)

            val bboxWidth = displayRight - displayLeft
            val bboxHeight = displayBottom - displayTop

            // Hanya gambar jika ukuran valid
            if (bboxWidth > 0 && bboxHeight > 0) {
                val cornerRadiusValue = 16f
                val cornerRadius = CornerRadius(cornerRadiusValue)
                val strokeWidth = calculateStrokeWidth(warningLevel, index == 0)

                // Gambar efek glow jika kritis atau deteksi utama
                drawGlowEffect(warningLevel, index == 0, displayLeft, displayTop, bboxWidth, bboxHeight, cornerRadiusValue, colorWithAlpha, animatedAlpha)

                // Gambar kotak utama
                drawRoundRect(
                    color = colorWithAlpha,
                    topLeft = Offset(displayLeft, displayTop),
                    size = Size(bboxWidth, bboxHeight),
                    cornerRadius = cornerRadius,
                    style = Stroke(width = strokeWidth)
                )

                // Gambar aksen sudut
                drawCornerAccents(displayLeft, displayTop, displayRight, displayBottom, 24f, strokeWidth + 2f, warningColor)

                // Gambar isian semi-transparan
                drawRoundRect(
                    color = colorWithAlpha.copy(alpha = 0.12f),
                    topLeft = Offset(displayLeft, displayTop),
                    size = Size(bboxWidth, bboxHeight),
                    cornerRadius = cornerRadius
                )

                // Gambar label nama, confidence, dan jarak
                drawDetectionLabel(index, displayName, detection.confidence, distInfo.distanceMeters, displayLeft, displayTop, warningColor, paintPool)

                // Gambar badge kategori jarak (hanya untuk 3 teratas)
                if (index < 3) {
                    drawDistanceBadge(distInfo, displayLeft, displayBottom, bboxWidth, paintPool)
                }

                // Gambar garis & titik penunjuk jarak (hanya untuk 3 teratas)
                if (index < 3) {
                    drawDistanceIndicator(canvasWidth, canvasHeight, displayLeft, displayTop, bboxWidth, bboxHeight, colorWithAlpha, warningColor)
                }
            }
        }
    }
}

/** Menggambar indikator saat tidak ada deteksi */
private fun DrawScope.drawScanningIndicator(canvasWidth: Float, canvasHeight: Float, paintPool: PaintPool) {
    // Garis pemindaian bergerak
    val scanLineY = (canvasHeight / 2f) + (canvasHeight / 4f) * sin(System.currentTimeMillis() / 500.0).toFloat()
    drawLine(
        color = Color(0xFF00BCD4).copy(alpha = 0.6f),
        start = Offset(0f, scanLineY),
        end = Offset(canvasWidth, scanLineY),
        strokeWidth = 3f
    )

    // Teks "Memindai..."
    val scanTextPaint = paintPool.getTextPaint().apply {
        color = Color.White.toArgb()
        textSize = 48f
        textAlign = Paint.Align.CENTER
    }
    drawContext.canvas.nativeCanvas.drawText(
        "Memindai Lingkungan...",
        canvasWidth / 2f,
        canvasHeight / 2f - 50f,
        scanTextPaint
    )
    paintPool.recyclePaint(scanTextPaint) // Kembalikan paint ke pool
}

/** Menghitung lebar garis bbox */
private fun calculateStrokeWidth(warningLevel: DepthEstimator.WarningLevel, isPrimary: Boolean): Float {
    return when (warningLevel) {
        DepthEstimator.WarningLevel.CRITICAL -> 8f
        DepthEstimator.WarningLevel.HIGH -> 6f
        else -> if (isPrimary) 6f else 4f
    }
}

/** Menggambar efek glow di sekitar bbox */
private fun DrawScope.drawGlowEffect(
    warningLevel: DepthEstimator.WarningLevel,
    isPrimary: Boolean,
    left: Float, top: Float, width: Float, height: Float,
    cornerRadius: Float, color: Color, animatedAlpha: Float
) {
    if (warningLevel == DepthEstimator.WarningLevel.CRITICAL) {
        drawRoundRect( // Glow merah untuk kritis
            color = Color.Red.copy(alpha = 0.3f * animatedAlpha),
            topLeft = Offset(left - 8f, top - 8f),
            size = Size(width + 16f, height + 16f),
            cornerRadius = CornerRadius(cornerRadius + 4f)
        )
    } else if (isPrimary) {
        drawRoundRect( // Glow tipis untuk deteksi utama
            color = color.copy(alpha = 0.15f),
            topLeft = Offset(left - 4f, top - 4f),
            size = Size(width + 8f, height + 8f),
            cornerRadius = CornerRadius(cornerRadius + 2f)
        )
    }
}

/** Menggambar aksen di sudut bbox */
private fun DrawScope.drawCornerAccents(
    left: Float, top: Float, right: Float, bottom: Float,
    length: Float, width: Float, color: Color
) {
    // Gunakan satu objek Path untuk efisiensi
    val path = Path()
    // Kiri atas
    path.moveTo(left, top + length); path.lineTo(left, top); path.lineTo(left + length, top)
    // Kanan atas
    path.moveTo(right - length, top); path.lineTo(right, top); path.lineTo(right, top + length)
    // Kanan bawah
    path.moveTo(right, bottom - length); path.lineTo(right, bottom); path.lineTo(right - length, bottom)
    // Kiri bawah
    path.moveTo(left + length, bottom); path.lineTo(left, bottom); path.lineTo(left, bottom - length)

    drawPath(path, color = color, style = Stroke(width = width))
}

/** Menggambar label deteksi (nama, confidence, jarak) */
private fun DrawScope.drawDetectionLabel(
    index: Int, displayName: String, confidence: Float, distance: Float,
    left: Float, top: Float, color: Color, paintPool: PaintPool
) {
    val rankBadge = if (index == 0) "★" else "#${index + 1}"
    val distanceText = String.format("%.1f", distance)
    val labelText = "$rankBadge $displayName ${(confidence * 100).toInt()}% • ${distanceText}m"

    val textPaint = paintPool.getTextPaint().apply {
        this.color = Color.White.toArgb()
        textSize = if (index == 0) 42f else 36f // Ukuran font beda untuk utama
        isAntiAlias = true
        isFakeBoldText = true
        setShadowLayer(4f, 0f, 2f, Color.Black.copy(alpha = 0.8f).toArgb())
    }

    val textWidth = textPaint.measureText(labelText)
    val textHeight = 56f
    val labelPadding = 16f

    // Pastikan label tidak keluar dari atas layar
    val labelTop = max(0f, top - textHeight - 8f)
    val labelLeft = left

    // Latar belakang label
    drawRoundRect(
        color = color.copy(alpha = if (index == 0) 1.0f else 0.95f),
        topLeft = Offset(labelLeft, labelTop),
        size = Size(textWidth + labelPadding * 2, textHeight),
        cornerRadius = CornerRadius(12f)
    )
    // Efek inner glow tipis
    drawRoundRect(
        color = Color.White.copy(alpha = 0.15f),
        topLeft = Offset(labelLeft + 2f, labelTop + 2f),
        size = Size(textWidth + labelPadding * 2 - 4f, textHeight - 4f),
        cornerRadius = CornerRadius(10f)
    )
    // Teks label
    drawContext.canvas.nativeCanvas.drawText(
        labelText,
        labelLeft + labelPadding,
        labelTop + textPaint.textSize * 0.9f + 4f, // Posisi Y disesuaikan agar pas tengah
        textPaint
    )
    paintPool.recyclePaint(textPaint) // Kembalikan paint
}

/** Menggambar badge kategori jarak */
private fun DrawScope.drawDistanceBadge(
    distInfo: DepthEstimator.DistanceInfo,
    left: Float, bottom: Float, bboxWidth: Float, paintPool: PaintPool
) {
    // Ambil hanya kata pertama (misal: "sangat dekat")
    val distanceBadgeText = distInfo.indonesianText.substringBefore(',').trim()
    if (distanceBadgeText.isEmpty()) return // Jangan gambar jika teks kosong

    val badgeTextPaint = paintPool.getTextPaint().apply {
        color = Color.White.toArgb()
        textSize = 32f
        isAntiAlias = true
        isFakeBoldText = true
        textAlign = Paint.Align.CENTER
        setShadowLayer(3f, 0f, 2f, Color.Black.copy(alpha = 0.7f).toArgb())
    }

    val badgeWidth = badgeTextPaint.measureText(distanceBadgeText) + 32f // Padding horizontal
    val badgeHeight = 44f
    // Posisi badge di bawah tengah bbox
    val badgeLeft = left + (bboxWidth - badgeWidth) / 2f
    val badgeTop = bottom + 8f

    // Latar belakang badge
    drawRoundRect(
        color = getDistanceBadgeColor(distInfo.category),
        topLeft = Offset(badgeLeft, badgeTop),
        size = Size(badgeWidth, badgeHeight),
        cornerRadius = CornerRadius(8f)
    )
    // Teks badge
    drawContext.canvas.nativeCanvas.drawText(
        distanceBadgeText,
        badgeLeft + badgeWidth / 2f,
        badgeTop + badgeHeight / 2f + badgeTextPaint.textSize / 3f, // Posisi Y agar pas tengah
        badgeTextPaint
    )
    paintPool.recyclePaint(badgeTextPaint) // Kembalikan paint
}

/** Menggambar garis dan titik indikator jarak */
private fun DrawScope.drawDistanceIndicator(
    canvasWidth: Float, canvasHeight: Float,
    left: Float, top: Float, width: Float, height: Float,
    lineColor: Color, pointColor: Color
) {
    val centerX = left + width / 2f
    val centerY = top + height / 2f

    // Garis dari bawah tengah layar ke pusat bbox
    drawLine(
        color = lineColor.copy(alpha = 0.4f),
        start = Offset(canvasWidth / 2f, canvasHeight),
        end = Offset(centerX, centerY),
        strokeWidth = 2f
    )
    // Lingkaran luar di pusat bbox
    drawCircle(color = pointColor, radius = 8f, center = Offset(centerX, centerY))
    // Lingkaran dalam putih
    drawCircle(color = Color.White, radius = 4f, center = Offset(centerX, centerY))
}

/** Pool untuk objek Paint agar tidak dibuat ulang */
private class PaintPool {
    private val pool = mutableListOf<Paint>()
    private val maxSize = 10 // Maksimal simpan 10 objek Paint

    @Synchronized
    fun getTextPaint(): Paint {
        return if (pool.isNotEmpty()) {
            pool.removeLast().apply { reset() } // Ambil dari belakang, reset
        } else {
            Paint().apply { isAntiAlias = true } // Buat baru jika pool kosong
        }
    }

    @Synchronized
    fun recyclePaint(paint: Paint) {
        if (pool.size < maxSize) {
            paint.reset() // Reset sebelum disimpan
            pool.add(paint) // Tambahkan ke pool
        }
        // Jika pool penuh, biarkan objek Paint dihancurkan oleh GC
    }
}

// Helper untuk mendapatkan warna dasar dan nama tampilan
private fun getColorAndName(className: String): Pair<Color, String> {
    return when (className.lowercase()) {
        "stair" -> Pair(Color(0xFFFF5252), "Tangga")
        "door" -> Pair(Color(0xFF448AFF), "Pintu")
        "person" -> Pair(Color(0xFF69F0AE), "Orang")
        "table" -> Pair(Color(0xFFE040FB), "Meja")
        "chair" -> Pair(Color(0xFFFDD835), "Kursi")
        else -> Pair(Color.LightGray, className.replaceFirstChar { it.titlecase() }) // Default abu-abu
    }
}

// Helper untuk mendapatkan warna peringatan
private fun getWarningColor(warningLevel: DepthEstimator.WarningLevel, baseColor: Color): Color {
    return when (warningLevel) {
        DepthEstimator.WarningLevel.CRITICAL -> Color(0xFFFF1744) // Merah terang
        DepthEstimator.WarningLevel.HIGH -> Color(0xFFFF9100) // Oranye terang
        DepthEstimator.WarningLevel.MEDIUM -> Color(0xFFFFC400) // Kuning terang
        DepthEstimator.WarningLevel.LOW -> baseColor // Gunakan warna dasar
    }
}

// Helper untuk mendapatkan warna badge jarak
private fun getDistanceBadgeColor(category: DepthEstimator.DistanceCategory): Color {
    val alpha = 0.9f
    return when (category) {
        DepthEstimator.DistanceCategory.VERY_CLOSE -> Color(0xFFD32F2F).copy(alpha = alpha) // Merah tua
        DepthEstimator.DistanceCategory.CLOSE -> Color(0xFFFF6F00).copy(alpha = alpha) // Oranye tua
        DepthEstimator.DistanceCategory.MEDIUM -> Color(0xFFFBC02D).copy(alpha = alpha) // Kuning tua
        DepthEstimator.DistanceCategory.FAR -> Color(0xFF388E3C).copy(alpha = alpha) // Hijau tua
        DepthEstimator.DistanceCategory.VERY_FAR -> Color(0xFF1976D2).copy(alpha = alpha) // Biru tua
    }
}