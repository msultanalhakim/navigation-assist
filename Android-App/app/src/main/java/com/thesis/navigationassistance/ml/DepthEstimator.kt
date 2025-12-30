package com.thesis.navigationassistance.ml

import android.content.Context
import android.util.Log
import com.thesis.navigationassistance.data.BoundingBox
import com.thesis.navigationassistance.data.Detection
import kotlin.math.max
import kotlin.math.min

/**
 * Depth Estimator - Jarak Dekat (0.5m - 8m)
 */
class DepthEstimator(private val context: Context) {

    companion object {
        private const val TAG = "DepthEstimator"

        // Parameter kamera (perlu dikalibrasi per perangkat)
        private const val FOCAL_LENGTH_MM = 4.25f
        private const val SENSOR_HEIGHT_MM = 5.76f
        private const val SENSOR_WIDTH_MM = 7.68f

        // Dimensi objek dunia nyata (meter) - asumsi
        private val OBJECT_REAL_HEIGHTS = mapOf(
            "person" to 1.65f,
            "door" to 2.10f,
            "chair" to 0.85f,
            "table" to 0.75f,
            "stair" to 0.18f // Tinggi satu anak tangga
        )

        private val OBJECT_REAL_WIDTHS = mapOf(
            "person" to 0.45f,
            "door" to 0.90f,
            "chair" to 0.50f,
            "table" to 1.20f,
            "stair" to 1.00f // Lebar tangga
        )

        // Ambang batas jarak (meter)
        const val VERY_CLOSE = 0.8f
        const val CLOSE = 1.5f
        const val MEDIUM = 3.0f
        const val FAR = 5.0f

        // Ambang batas cakupan layar
        private const val EXTREME_CLOSE_COVERAGE = 0.85f
        private const val FULL_SCREEN_COVERAGE = 0.70f
        private const val HIGH_COVERAGE = 0.50f
        private const val MEDIUM_COVERAGE = 0.30f
        private const val LOW_COVERAGE = 0.12f

        private const val MIN_BBOX_SIZE = 15f // Ukuran minimum bbox dalam pixel
        private const val MAX_RELIABLE_DISTANCE = 8.0f // Jarak maksimum yang dianggap andal
        private const val MIN_RELIABLE_DISTANCE = 0.4f // Jarak minimum yang dianggap andal

        // Tabel lookup kalibrasi (faktor pengali untuk koreksi)
        private val CALIBRATION_TABLE = floatArrayOf(
            1.28f, // 0.0-0.5m
            1.18f, // 0.5-0.7m
            1.12f, // 0.7-1.0m
            1.05f, // 1.0-1.5m
            1.00f, // 1.5-2.5m (baseline)
            0.95f, // 2.5-4.0m
            0.90f, // 4.0-6.0m
            0.85f  // 6.0m+
        )

        // Konstanta pra-hitung untuk pinhole
        private const val FOCAL_HEIGHT_PORTRAIT = FOCAL_LENGTH_MM / SENSOR_HEIGHT_MM
        private const val FOCAL_WIDTH_PORTRAIT = FOCAL_LENGTH_MM / SENSOR_WIDTH_MM
        private const val FOCAL_HEIGHT_LANDSCAPE = FOCAL_LENGTH_MM / SENSOR_WIDTH_MM
        private const val FOCAL_WIDTH_LANDSCAPE = FOCAL_LENGTH_MM / SENSOR_HEIGHT_MM
    }

    data class DistanceInfo(
        val distanceMeters: Float,
        val category: DistanceCategory,
        val confidence: Float, // Seberapa yakin estimator dengan hasil ini
        val method: EstimationMethod,
        val indonesianText: String
    )

    enum class DistanceCategory {
        VERY_CLOSE, CLOSE, MEDIUM, FAR, VERY_FAR
    }

    enum class EstimationMethod {
        SCREEN_COVERAGE, // Berdasarkan persentase layar yang tertutup
        KNOWN_SIZE,      // Berdasarkan ukuran objek nyata (pinhole)
        HYBRID           // Kombinasi keduanya
    }

    enum class WarningLevel {
        LOW, MEDIUM, HIGH, CRITICAL
    }

    // Cache untuk menghindari pemformatan string berulang
    private val distanceTextCache = mutableMapOf<Pair<Float, DistanceCategory>, String>()

    /**
     * Estimasi jarak utama dengan pemeriksaan kewajaran
     */
    fun estimateDistance(
        detection: Detection,
        imageWidth: Int,
        imageHeight: Int
    ): DistanceInfo {
        val className = detection.className.lowercase()
        val bbox = detection.bbox

        // Validasi dimensi gambar
        if (imageWidth <= 0 || imageHeight <= 0) {
            Log.e(TAG, "Invalid image dimensions: ${imageWidth}x${imageHeight}")
            return createFallbackDistanceInfo()
        }

        // Validasi dimensi bbox
        if (!bbox.isValid()) { // Menggunakan validasi dari BoundingBox
            Log.e(TAG, "Invalid bbox dimensions: ${bbox.width}x${bbox.height}")
            return createFallbackDistanceInfo()
        }

        // Pra-hitung nilai umum
        val isPortrait = imageHeight > imageWidth
        val imageArea = imageWidth * imageHeight.toFloat()

        if (imageArea <= 0f) {
            return createFallbackDistanceInfo()
        }

        val bboxArea = bbox.area
        val screenCoverage = if (imageArea > 0f) bboxArea / imageArea else 0f
        val verticalCoverage = if (imageHeight > 0) bbox.height / imageHeight else 0f
        val horizontalCoverage = if (imageWidth > 0) bbox.width / imageWidth else 0f
        // Cakupan utama: vertikal jika portrait, horizontal jika landscape
        val primaryCoverage = if (isPortrait) verticalCoverage else horizontalCoverage

        Log.d(TAG, "Distance Estimation: $className")
        Log.d(TAG, "   Image: ${imageWidth}x${imageHeight} (${if (isPortrait) "Portrait" else "Landscape"})")
        Log.d(TAG, "   Coverage: ${(screenCoverage * 100).toInt()}%")

        // Pemeriksaan kewajaran aspect ratio
        val aspectRatio = bbox.aspectRatio
        val expectedAspectRatio = getExpectedAspectRatio(className)

        // Penalti confidence jika AR mencurigakan
        val confidencePenalty = if (aspectRatio !in expectedAspectRatio) {
            Log.w(TAG, "   WARNING: Suspicious AR ${String.format("%.2f", aspectRatio)} for $className (Expected: $expectedAspectRatio)")
            0.6f // Penalti 40%
        } else {
            1.0f // Tidak ada penalti
        }

        // Metode 1: Pinhole (menggunakan ukuran objek nyata)
        val pinholeDistance = estimateUsingOptimizedPinhole(
            bbox, className, imageWidth, imageHeight, isPortrait
        )

        // Metode 2: Cakupan Layar
        val coverageDistance = estimateFromCoverageFast(
            screenCoverage, primaryCoverage, className
        )

        // Metode 3: Pilih estimasi terbaik
        val (finalDistance, method, confidence) = selectBestEstimate(
            pinholeDistance,
            coverageDistance,
            screenCoverage,
            className
        )

        // Kategorisasi jarak dan buat teks
        val category = categorizeDistanceFast(finalDistance)
        val indonesianText = getIndonesianDistanceCached(finalDistance, category)

        Log.d(TAG, "   Final Distance: ${String.format("%.2f", finalDistance)}m")
        Log.d(TAG, "   Method: $method | Confidence: ${(confidence * 100).toInt()}%")
        if (confidencePenalty < 1.0f) {
            Log.d(TAG, "   Confidence penalty applied due to AR mismatch.")
        }

        // Gabungkan confidence estimator dengan confidence deteksi awal dan penalti AR
        val combinedConfidence = (confidence * detection.confidence * confidencePenalty).coerceIn(0f, 1f)

        return DistanceInfo(
            distanceMeters = finalDistance,
            category = category,
            confidence = combinedConfidence,
            method = method,
            indonesianText = indonesianText
        )
    }

    /**
     * Estimasi pinhole dengan konstanta pra-hitung & proteksi pembagian nol
     */
    private fun estimateUsingOptimizedPinhole(
        bbox: BoundingBox,
        className: String,
        imageWidth: Int,
        imageHeight: Int,
        isPortrait: Boolean
    ): Float? {
        val realHeight = OBJECT_REAL_HEIGHTS[className]
        val realWidth = OBJECT_REAL_WIDTHS[className]

        // Jika tidak tahu ukuran objek, metode ini tidak bisa digunakan
        if (realHeight == null && realWidth == null) return null

        val bboxHeightPx = bbox.height
        val bboxWidthPx = bbox.width

        // Periksa ukuran minimum bbox
        if (bboxHeightPx < MIN_BBOX_SIZE && bboxWidthPx < MIN_BBOX_SIZE) return null

        // Gunakan rasio fokal pra-hitung
        val distanceFromHeight = realHeight?.let {
            if (bboxHeightPx >= MIN_BBOX_SIZE) {
                // Dimensi gambar yang relevan (tinggi untuk portrait, lebar untuk landscape)
                val imageDim = if (isPortrait) imageHeight.toFloat() else imageWidth.toFloat()
                val focalRatio = if (isPortrait) FOCAL_HEIGHT_PORTRAIT else FOCAL_HEIGHT_LANDSCAPE
                // Proteksi pembagian nol
                if (bboxHeightPx > 0f) (it * focalRatio * imageDim) / bboxHeightPx else null
            } else null
        }

        val distanceFromWidth = realWidth?.let {
            if (bboxWidthPx >= MIN_BBOX_SIZE) {
                // Dimensi gambar yang relevan (lebar untuk portrait, tinggi untuk landscape)
                val imageDim = if (isPortrait) imageWidth.toFloat() else imageHeight.toFloat()
                val focalRatio = if (isPortrait) FOCAL_WIDTH_PORTRAIT else FOCAL_WIDTH_LANDSCAPE
                // Proteksi pembagian nol
                if (bboxWidthPx > 0f) (it * focalRatio * imageDim) / bboxWidthPx else null
            } else null
        }

        // Rata-rata jika keduanya tersedia, jika tidak gunakan yang ada
        val rawDistance = when {
            distanceFromHeight != null && distanceFromWidth != null -> (distanceFromHeight + distanceFromWidth) * 0.5f
            distanceFromHeight != null -> distanceFromHeight
            distanceFromWidth != null -> distanceFromWidth
            else -> return null // Keduanya gagal
        }

        // Terapkan faktor kalibrasi dari tabel lookup
        val calibrationFactor = getCalibrationFactor(rawDistance)
        val distance = rawDistance * calibrationFactor

        Log.d(TAG, "   Pinhole Estimate: Raw=${String.format("%.2f", rawDistance)}m -> Calibrated=${String.format("%.2f", distance)}m (Factor: ${calibrationFactor})")

        // Kembalikan hanya jika dalam rentang andal
        return if (distance in MIN_RELIABLE_DISTANCE..MAX_RELIABLE_DISTANCE) {
            distance
        } else {
            Log.d(TAG, "   Pinhole result discarded (outside reliable range).")
            null
        }
    }

    /**
     * Lookup faktor kalibrasi cepat
     */
    private fun getCalibrationFactor(distance: Float): Float {
        return when {
            distance < 0.5f -> CALIBRATION_TABLE[0]
            distance < 0.7f -> CALIBRATION_TABLE[1]
            distance < 1.0f -> CALIBRATION_TABLE[2]
            distance < 1.5f -> CALIBRATION_TABLE[3]
            distance < 2.5f -> CALIBRATION_TABLE[4]
            distance < 4.0f -> CALIBRATION_TABLE[5]
            distance < 6.0f -> CALIBRATION_TABLE[6]
            else -> CALIBRATION_TABLE[7]
        }
    }

    /**
     * Estimasi berbasis cakupan (perhitungan disederhanakan) + proteksi nol
     */
    private fun estimateFromCoverageFast(
        screenCoverage: Float,
        primaryCoverage: Float,
        className: String
    ): Float? {
        // Metode ini membutuhkan tinggi objek nyata
        val realHeight = OBJECT_REAL_HEIGHTS[className] ?: return null

        // Proteksi nol/negatif
        if (primaryCoverage <= 0f || screenCoverage <= 0f) return null

        // Estimasi kasar berdasarkan cakupan
        val distance = when {
            primaryCoverage > 0.90f || screenCoverage > EXTREME_CLOSE_COVERAGE -> (realHeight / primaryCoverage * 2.8f).coerceIn(0.4f, 0.7f)
            primaryCoverage > 0.75f || screenCoverage > FULL_SCREEN_COVERAGE -> (realHeight / primaryCoverage * 2.5f).coerceIn(0.5f, 0.9f)
            primaryCoverage > 0.55f || screenCoverage > HIGH_COVERAGE -> (realHeight / primaryCoverage * 2.0f).coerceIn(0.8f, 1.8f)
            primaryCoverage > 0.30f || screenCoverage > MEDIUM_COVERAGE -> (realHeight / primaryCoverage * 1.6f).coerceIn(1.5f, 3.5f)
            primaryCoverage > 0.15f || screenCoverage > LOW_COVERAGE -> (realHeight / primaryCoverage * 1.2f).coerceIn(3.0f, 6.0f)
            else -> {
                // Jika cakupan sangat kecil, mungkin tidak andal
                if (primaryCoverage < 0.08f) return null
                (realHeight / primaryCoverage * 0.9f).coerceIn(5.0f, MAX_RELIABLE_DISTANCE)
            }
        }

        Log.d(TAG, "   Coverage Estimate: ${String.format("%.2f", distance)}m (Screen: ${(screenCoverage*100).toInt()}%, Primary: ${(primaryCoverage*100).toInt()}%)")
        return distance
    }

    /**
     * Pemilihan estimasi terbaik yang disederhanakan
     */
    private fun selectBestEstimate(
        pinholeDistance: Float?,
        coverageDistance: Float?,
        screenCoverage: Float,
        className: String
    ): Triple<Float, EstimationMethod, Float> { // (Jarak, Metode, Confidence Estimator)

        // Prioritas: Metode Pinhole jika tersedia
        if (pinholeDistance != null) {
            // Jika metode cakupan juga tersedia, cek konsistensi
            if (coverageDistance != null) {
                val diff = kotlin.math.abs(pinholeDistance - coverageDistance)
                val avgDistance = (pinholeDistance + coverageDistance) * 0.5f

                // Jika perbedaannya kecil (<25% dari rata-rata), gabungkan (hybrid)
                if (diff < avgDistance * 0.25f) {
                    val blended = pinholeDistance * 0.75f + coverageDistance * 0.25f // Lebih condong ke pinhole
                    return Triple(blended, EstimationMethod.HYBRID, 0.92f) // Confidence tinggi jika konsisten
                }
            }
            // Jika tidak ada cakupan atau tidak konsisten, gunakan pinhole saja
            return Triple(pinholeDistance, EstimationMethod.KNOWN_SIZE, 0.90f)
        }

        // Fallback: Metode Cakupan jika Pinhole gagal
        if (coverageDistance != null) {
            // Confidence metode cakupan lebih rendah, tergantung seberapa besar cakupannya
            val confidence = when {
                screenCoverage > HIGH_COVERAGE -> 0.80f
                screenCoverage > MEDIUM_COVERAGE -> 0.70f
                else -> 0.60f
            }
            return Triple(coverageDistance, EstimationMethod.SCREEN_COVERAGE, confidence)
        }

        // Pilihan Terakhir: Estimasi empiris kasar berdasarkan cakupan layar
        Log.w(TAG, "   Both estimation methods failed, using empirical fallback.")
        val fallbackDistance = when {
            screenCoverage > 0.80f -> 0.5f
            screenCoverage > 0.65f -> 0.7f
            screenCoverage > 0.50f -> 0.9f
            screenCoverage > 0.35f -> 1.3f
            screenCoverage > 0.20f -> 2.5f
            screenCoverage > 0.10f -> 4.0f
            else -> 6.0f // Jika sangat kecil, anggap jauh
        }
        return Triple(fallbackDistance, EstimationMethod.SCREEN_COVERAGE, 0.50f) // Confidence rendah untuk fallback
    }

    /**
     * Kategorisasi jarak cepat
     */
    private fun categorizeDistanceFast(distance: Float): DistanceCategory {
        return when {
            distance < VERY_CLOSE -> DistanceCategory.VERY_CLOSE
            distance < CLOSE -> DistanceCategory.CLOSE
            distance < MEDIUM -> DistanceCategory.MEDIUM
            distance < FAR -> DistanceCategory.FAR
            else -> DistanceCategory.VERY_FAR
        }
    }

    /**
     * Pembuatan teks Indonesia dengan cache
     */
    private fun getIndonesianDistanceCached(distance: Float, category: DistanceCategory): String {
        val cacheKey = Pair(distance, category)
        // Ambil dari cache jika ada, jika tidak, buat dan simpan
        return distanceTextCache.getOrPut(cacheKey) {
            generateIndonesianText(distance, category)
        }
    }

    /**
     * Buat teks jarak Indonesia
     */
    private fun generateIndonesianText(distance: Float, category: DistanceCategory): String {
        val distanceText = when (category) {
            DistanceCategory.VERY_CLOSE -> "sangat dekat"
            DistanceCategory.CLOSE -> "dekat"
            DistanceCategory.MEDIUM -> "sedang"
            DistanceCategory.FAR -> "jauh"
            DistanceCategory.VERY_FAR -> "sangat jauh"
        }

        // Format angka: cm jika < 1m, 1 desimal jika < 10m, bulat jika >= 10m
        val meters = when {
            distance < 1.0f -> "${(distance * 100).toInt()} sentimeter"
            distance < 10.0f -> String.format("%.1f", distance).replace('.', ',') + " meter"
            else -> "${distance.toInt()} meter"
        }

        return "$distanceText, $meters"
    }

    /**
     * Perhitungan skor keselamatan cepat
     */
    fun calculateSafetyScore(
        detection: Detection,
        distanceInfo: DistanceInfo
    ): Int {
        // Skor dasar berdasarkan kategori jarak (0 = sangat bahaya, 100 = aman)
        val distanceScore = when (distanceInfo.category) {
            DistanceCategory.VERY_CLOSE -> 15
            DistanceCategory.CLOSE -> 45
            DistanceCategory.MEDIUM -> 70
            DistanceCategory.FAR -> 85
            DistanceCategory.VERY_FAR -> 95
        }

        // Faktor bahaya intrinsik objek (tangga lebih bahaya dari pintu)
        val objectDangerFactor = when (detection.className.lowercase()) {
            "stair" -> 0.65f // Tangga mengurangi skor keselamatan
            "person" -> 0.85f // Orang sedikit mengurangi
            "door" -> 1.0f    // Pintu dianggap netral
            "table", "chair" -> 0.80f // Meja/kursi cukup bahaya
            else -> 0.90f // Objek lain sedikit mengurangi
        }

        // Skor akhir = skor jarak * faktor bahaya objek
        return (distanceScore * objectDangerFactor).toInt().coerceIn(0, 100)
    }

    /**
     * Penentuan level peringatan cepat
     */
    fun getWarningLevel(
        detection: Detection,
        distanceInfo: DistanceInfo
    ): WarningLevel {
        val safetyScore = calculateSafetyScore(detection, distanceInfo)
        val className = detection.className.lowercase()

        // Tangga punya ambang batas lebih ketat
        if (className == "stair") {
            return when {
                safetyScore < 25 -> WarningLevel.CRITICAL
                safetyScore < 45 -> WarningLevel.HIGH
                safetyScore < 65 -> WarningLevel.MEDIUM
                else -> WarningLevel.LOW
            }
        }

        // Ambang batas umum untuk objek lain
        return when {
            safetyScore < 30 -> WarningLevel.CRITICAL
            safetyScore < 50 -> WarningLevel.HIGH
            safetyScore < 70 -> WarningLevel.MEDIUM
            else -> WarningLevel.LOW
        }
    }

    /**
     * Bersihkan cache (panggil secara berkala untuk manajemen memori)
     */
    fun clearCache() {
        // Hanya bersihkan jika cache terlalu besar
        if (distanceTextCache.size > 100) {
            distanceTextCache.clear()
            Log.d(TAG, "Distance text cache cleared.")
        }
    }

    /**
     * Hasil fallback untuk kasus error
     */
    private fun createFallbackDistanceInfo(): DistanceInfo {
        return DistanceInfo(
            distanceMeters = MAX_RELIABLE_DISTANCE, // Anggap jauh dan aman
            category = DistanceCategory.VERY_FAR,
            confidence = 0.0f, // Confidence nol
            method = EstimationMethod.SCREEN_COVERAGE, // Metode fallback
            indonesianText = "sangat jauh, tidak dapat diukur"
        )
    }

    /**
     * Helper untuk mendapatkan rentang AR yang diharapkan per kelas
     */
    private fun getExpectedAspectRatio(className: String) : ClosedFloatingPointRange<Float>{
        return when (className) {
            "chair" -> 0.35f..1.70f // Bisa tinggi (sandaran) atau lebar (bangku)
            "table" -> 0.25f..5.0f // Bisa sangat lebar
            "door" -> 0.15f..0.55f // Biasanya tinggi dan kurus
            "person" -> 0.20f..0.65f // Biasanya tinggi
            "stair" -> 0.50f..4.5f // Bisa lebar (tangga lurus) atau tinggi (jika hanya 1 anak tangga terlihat)
            else -> 0.1f..10.0f // Rentang default sangat lebar
        }
    }
}