package com.thesis.navigationassistance.viewmodel

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.thesis.navigationassistance.data.Detection
import com.thesis.navigationassistance.data.DetectionWithDistance
import com.thesis.navigationassistance.ml.YoloDetector
import com.thesis.navigationassistance.ml.DepthEstimator
import com.thesis.navigationassistance.ml.PersistentObjectTracker
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

class DetectionViewModel(application: Application) : AndroidViewModel(application) {

    private val _detections = MutableStateFlow<List<Detection>>(emptyList())
    val detections: StateFlow<List<Detection>> = _detections.asStateFlow()

    private val _detectionsWithDistance = MutableStateFlow<List<DetectionWithDistance>>(emptyList())
    val detectionsWithDistance: StateFlow<List<DetectionWithDistance>> = _detectionsWithDistance.asStateFlow()

    private val _isProcessing = MutableStateFlow(false)
    val isProcessing: StateFlow<Boolean> = _isProcessing.asStateFlow()

    private val _fps = MutableStateFlow(0f)
    val fps: StateFlow<Float> = _fps.asStateFlow()

    private val _screenWidth = MutableStateFlow(1080) // Default, akan diupdate
    val screenWidth: StateFlow<Int> = _screenWidth.asStateFlow()

    // State dan fungsi terkait debug dihapus

    private var currentImageWidth = 0
    private var currentImageHeight = 0

    private var tts: TextToSpeech? = null
    private var vibrator: Vibrator? = null
    private lateinit var detector: YoloDetector
    private lateinit var depthEstimator: DepthEstimator
    private lateinit var persistentTracker: PersistentObjectTracker

    private val isProcessingFrame = AtomicBoolean(false)
    private var processingJob: Job? = null

    private val lastAnnouncedObjects = mutableMapOf<String, Long>()
    private val announceInterval = 800L
    private var lastFrameTime = System.currentTimeMillis()
    private var lastDetectionTime = 0L
    private val detectionCooldown = 25L // Cooldown bisa disesuaikan, misal 20L atau 30L
    private var isSpeaking = false

    private val stairTracker = mutableMapOf<String, StairTracker>()
    private val otherTracker = mutableMapOf<String, DetectionTracker>()

    companion object {
        private const val TAG = "DetectionViewModel"

        // Threshold diturunkan sedikit
        private const val CONFIDENCE_THRESHOLD = 0.30f // Dari 0.32f
        private const val ANNOUNCE_CONFIDENCE = 0.42f
        private const val IOU_THRESHOLD = 0.40f

        private const val STAIR_TRACKING_THRESHOLD = 2
        private const val CLOSE_OBJECT_THRESHOLD = 2
        private const val OTHER_TRACKING_THRESHOLD = 3

        private val CLASS_CONFIDENCE_OVERRIDE = mapOf(
            "stair" to 0.45f,
            "person" to 0.55f,
            "door" to 0.50f,
            "table" to 0.50f,
            "chair" to 0.70f
        )

        private val ANNOUNCEMENT_PRIORITY = listOf(
            "stair", "person", "door", "table", "chair"
        )
    }

    private data class DetectionTracker(
        var lastSeen: Long = System.currentTimeMillis(),
        var consecutiveFrames: Int = 1,
        var avgConfidence: Float = 0f,
        var lastBbox: com.thesis.navigationassistance.data.BoundingBox? = null,
        var stable: Boolean = false
        // lastDistance dihapus karena tidak dipakai
    )

    private data class StairTracker(
        var lastSeen: Long = System.currentTimeMillis(),
        var consecutiveFrames: Int = 1,
        var maxConfidence: Float = 0f,
        var lastBbox: com.thesis.navigationassistance.data.BoundingBox? = null,
        var stable: Boolean = false
        // lastDistance dihapus karena tidak dipakai
    )

    init {
        initializeServices()
    }

    private fun initializeServices() {
        tts = TextToSpeech(getApplication()) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val localeID = Locale("id", "ID")
                val result = tts?.setLanguage(localeID)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.w(TAG, "Bahasa Indonesia tidak tersedia, menggunakan US English.")
                    tts?.setLanguage(Locale.US)
                } else {
                    Log.d(TAG, "TTS Bahasa Indonesia siap.")
                    tts?.setSpeechRate(1.02f); tts?.setPitch(1.0f)
                }
                tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) { isSpeaking = true }
                    override fun onDone(utteranceId: String?) { isSpeaking = false }
                    override fun onError(utteranceId: String?) { isSpeaking = false }
                })
            } else { Log.e(TAG, "Inisialisasi TTS gagal!") }
        }

        vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vm = getApplication<Application>().getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            vm.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getApplication<Application>().getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }

        detector = YoloDetector(getApplication(), useGpu = true)
        depthEstimator = DepthEstimator(getApplication())
        persistentTracker = PersistentObjectTracker()

        Log.i(TAG, "DetectionViewModel initialized.")
        Log.d(TAG, "   ${detector.getAccelerationInfo()}")
        Log.d(TAG, "   Cooldown: ${detectionCooldown}ms")
    }

    fun setScreenWidth(width: Int) { _screenWidth.value = width }

    // Fungsi setDebugMode dihapus

    fun processImageAsync(bitmap: Bitmap) {
        if (!isProcessingFrame.compareAndSet(false, true)) {
            bitmap.recycleSafely(); return
        }
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastDetectionTime < detectionCooldown) {
            isProcessingFrame.set(false); bitmap.recycleSafely(); return
        }
        processingJob?.cancel()
        processingJob = viewModelScope.launch(Dispatchers.Default) {
            try {
                if (!isActive) { bitmap.recycleSafely(); return@launch }
                processImageInternal(bitmap, currentTime)
            } catch (e: Exception) {
                Log.e(TAG, "Error dalam pemrosesan gambar", e); bitmap.recycleSafely()
            } finally {
                isProcessingFrame.set(false)
            }
        }
    }

    private suspend fun processImageInternal(bitmap: Bitmap, currentTime: Long) {
        withContext(Dispatchers.Main) { _isProcessing.value = true }
        var processedBitmap: Bitmap? = bitmap
        try {
            val startTime = System.currentTimeMillis()
            val inputWidth = processedBitmap!!.width
            val inputHeight = processedBitmap!!.height
            currentImageWidth = inputWidth; currentImageHeight = inputHeight

            // Selalu panggil detect biasa (tanpa debug)
            val detectionResult = withContext(Dispatchers.Default) {
                detector.detect(
                    bitmap = processedBitmap!!,
                    confidenceThreshold = CONFIDENCE_THRESHOLD, // Gunakan threshold baru
                    iouThreshold = IOU_THRESHOLD
                )
            }

            processedBitmap?.recycleSafely(); processedBitmap = null

            val validResults = validateDetections(detectionResult)
            val trackingResult = persistentTracker.trackFrame(validResults, currentTime, currentImageWidth, currentImageHeight)
            if (trackingResult.lostTracking.isNotEmpty()) {
                handleVeryCloseObjects(trackingResult.lostTracking)
            }
            val stableResults = filterWithDualTracking(trackingResult.continuedTracking, currentTime)
            withContext(Dispatchers.Main) { _detections.value = stableResults }
            val resultsWithDistance = estimateDistancesParallel(stableResults)
            withContext(Dispatchers.Main) { _detectionsWithDistance.value = resultsWithDistance }
            updateFps(currentTime)
            lastDetectionTime = currentTime
            val processingTime = System.currentTimeMillis() - startTime
            Log.d(TAG, "Frame processed in ${processingTime}ms | FPS: ${_fps.value.toInt()}")
            if (resultsWithDistance.isNotEmpty()) {
                logDetectionResults(resultsWithDistance)
                val highConfDetections = filterAndSortDetectionsForAnnouncement(resultsWithDistance)
                if (highConfDetections.isNotEmpty() && !isSpeaking) {
                    announceDetections(highConfDetections, currentTime)
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error in processImageInternal", e); processedBitmap?.recycleSafely()
        } finally {
            withContext(Dispatchers.Main) { _isProcessing.value = false }
        }
    }

    // Fungsi logDebugInfo dihapus

    private suspend fun estimateDistancesParallel(detections: List<Detection>): List<DetectionWithDistance> {
        return withContext(Dispatchers.Default) {
            detections.map { detection ->
                async {
                    val distanceInfo = depthEstimator.estimateDistance(detection, currentImageWidth, currentImageHeight)
                    val safetyScore = depthEstimator.calculateSafetyScore(detection, distanceInfo)
                    val warningLevel = depthEstimator.getWarningLevel(detection, distanceInfo)
                    DetectionWithDistance(detection, distanceInfo, safetyScore, warningLevel)
                }
            }.map { it.await() }
        }
    }

    private fun updateFps(currentTime: Long) {
        val timeDiff = currentTime - lastFrameTime
        if (timeDiff > 0) {
            val instantFps = 1000f / timeDiff
            val newFps = if (_fps.value == 0f) instantFps else _fps.value * 0.85f + instantFps * 0.15f
            // Update StateFlow di Main thread
            viewModelScope.launch(Dispatchers.Main) { _fps.value = newFps }
        }
        lastFrameTime = currentTime
    }


    private fun validateDetections(results: List<Detection>): List<Detection> {
        if (currentImageWidth <= 0 || currentImageHeight <= 0) return emptyList()
        val imageArea = currentImageWidth * currentImageHeight.toFloat()
        if (imageArea <= 0f) return emptyList()
        return results.filter { detection ->
            val bbox = detection.bbox
            if (!bbox.isValid()) return@filter false
            val coverage = bbox.area / imageArea
            val margin = if (coverage > 0.60f) 30f else 10f
            bbox.x1 >= -margin && bbox.y1 >= -margin && bbox.x2 <= currentImageWidth + margin && bbox.y2 <= currentImageHeight + margin
        }
    }

    private fun handleVeryCloseObjects(lostObjects: List<PersistentObjectTracker.LostObjectInfo>) {
        lostObjects.forEach { lostObj ->
            if (!lostObj.shouldAnnounce) return@forEach
            val objectNameIndo = translateObjectName(lostObj.className)
            val distanceMeters = lostObj.estimatedDistance
            val distanceText = formatDistanceText(distanceMeters)
            val message = when (lostObj.className.lowercase()) {
                "stair" -> "Awas! Tangga sudah sangat dekat, kurang dari $distanceText!"
                "door" -> "Pintu sudah sangat dekat, kurang dari $distanceText!"
                "person" -> "Terdapat orang sangat dekat di depan Anda!"
                "table" -> "Meja sangat dekat, kurang dari $distanceText!"
                "chair" -> "Terdapat Kursi sangat dekat di depan Anda!"
                else -> "Awas! Ada $objectNameIndo sangat dekat di depan Anda!"
            }
            vibrateBasedOnDistance(DepthEstimator.DistanceCategory.VERY_CLOSE) // Akses via class
            speak(message, true)
            Log.w(TAG, "VERY CLOSE Obstacle Alert: $message")
        }
    }

    private fun formatDistanceText(distanceMeters: Float): String {
        return when {
            distanceMeters < 1.0f -> "${(distanceMeters * 100).toInt()} sentimeter"
            distanceMeters < 10.0f -> String.format("%.1f", distanceMeters).replace('.', ',') + " meter"
            else -> "${distanceMeters.toInt()} meter"
        }
    }

    fun resetTracking() {
        persistentTracker.reset(); stairTracker.clear(); otherTracker.clear(); lastAnnouncedObjects.clear()
        Log.i(TAG, "Tracking state reset.")
    }

    private fun filterWithDualTracking(detections: List<Detection>, currentTime: Long): List<Detection> {
        val stableDetections = mutableListOf<Detection>()
        if (currentImageWidth <= 0 || currentImageHeight <= 0) return emptyList()
        val imageArea = currentImageWidth * currentImageHeight.toFloat(); if (imageArea <= 0f) return emptyList()

        detections.forEach { det ->
            val gridX = (det.bbox.centerX / 80).toInt(); val gridY = (det.bbox.centerY / 80).toInt()
            val key = "${det.className}_${gridX}_${gridY}"
            val isStair = det.className.lowercase() in listOf("stair", "stairs")
            val coverage = det.bbox.area / imageArea
            val isVeryCloseCoverage = coverage > 0.60f

            if (isStair) {
                val tracker = stairTracker.getOrPut(key) { StairTracker() }
                updateStairTracker(tracker, det, currentTime, isVeryCloseCoverage)
                val minConfidenceStair = if (isVeryCloseCoverage) 0.35f else 0.40f
                if (tracker.stable || det.confidence >= minConfidenceStair) stableDetections.add(det)
            } else {
                val tracker = otherTracker.getOrPut(key) { DetectionTracker() }
                updateOtherTracker(tracker, det, currentTime, isVeryCloseCoverage)
                val minConfidenceOther = if (isVeryCloseCoverage) 0.45f else 0.65f
                if (tracker.stable || det.confidence >= minConfidenceOther) stableDetections.add(det)
            }
        }
        stairTracker.entries.removeIf { (_, t) -> currentTime - t.lastSeen > 400L }
        otherTracker.entries.removeIf { (_, t) -> currentTime - t.lastSeen > 500L }
        return stableDetections
    }

    private fun updateStairTracker(tracker: StairTracker, det: Detection, currentTime: Long, isVeryClose: Boolean) {
        val isSimilarPosition = tracker.lastBbox?.let { det.bbox.distanceTo(it) < if (isVeryClose) 250f else 200f } ?: true
        if (isSimilarPosition) {
            tracker.lastSeen = currentTime; tracker.consecutiveFrames++; tracker.maxConfidence = maxOf(tracker.maxConfidence, det.confidence)
            tracker.lastBbox = det.bbox; if (tracker.consecutiveFrames >= STAIR_TRACKING_THRESHOLD) tracker.stable = true
        } else {
            tracker.lastSeen = currentTime; tracker.consecutiveFrames = 1; tracker.maxConfidence = det.confidence
            tracker.lastBbox = det.bbox; tracker.stable = false
        }
    }

    private fun updateOtherTracker(tracker: DetectionTracker, det: Detection, currentTime: Long, isVeryClose: Boolean) {
        val isSimilarPosition = tracker.lastBbox?.let { det.bbox.distanceTo(it) < if (isVeryClose) 200f else 150f } ?: true
        if (isSimilarPosition) {
            tracker.lastSeen = currentTime; tracker.consecutiveFrames++
            tracker.avgConfidence = tracker.avgConfidence * 0.7f + det.confidence * 0.3f; tracker.lastBbox = det.bbox
            val requiredFrames = if (isVeryClose) CLOSE_OBJECT_THRESHOLD else OTHER_TRACKING_THRESHOLD
            if (tracker.consecutiveFrames >= requiredFrames) tracker.stable = true
        } else {
            tracker.lastSeen = currentTime; tracker.consecutiveFrames = 1; tracker.avgConfidence = det.confidence
            tracker.lastBbox = det.bbox; tracker.stable = false
        }
    }

    private fun logDetectionResults(results: List<DetectionWithDistance>) {
        Log.d(TAG, "--- Detection Results (${results.size}) ---")
        val stairs = results.filter { it.detection.className.lowercase() == "stair" }
        val veryCloseObjects = results.filter { it.distanceInfo.category == DepthEstimator.DistanceCategory.VERY_CLOSE }
        stairs.takeIf { it.isNotEmpty() }?.let {
            Log.d(TAG, "   Stairs Detected:")
            it.sortedBy { it.distanceInfo.distanceMeters }.forEach { det ->
                Log.d(TAG, "      - Tangga: ${det.detection.confidencePercent}% | ${String.format("%.1f", det.distanceInfo.distanceMeters)}m | Pos: ${getObjectPosition(det.detection)} | Level: ${det.warningLevel}")
            }
        }
        veryCloseObjects.takeIf { it.isNotEmpty() }?.let {
            Log.d(TAG, "   Very Close Objects (<${DepthEstimator.VERY_CLOSE}m):")
            it.sortedBy { it.distanceInfo.distanceMeters }.forEach { det ->
                Log.d(TAG, "      - ${translateObjectName(det.detection.className)}: ${det.detection.confidencePercent}% | ${String.format("%.1f", det.distanceInfo.distanceMeters)}m | Pos: ${getObjectPosition(det.detection)} | Level: ${det.warningLevel}")
            }
        }
    }

    private fun filterAndSortDetectionsForAnnouncement(results: List<DetectionWithDistance>): List<DetectionWithDistance> {
        return results
            .filter {
                val threshold = CLASS_CONFIDENCE_OVERRIDE[it.detection.className.lowercase()] ?: ANNOUNCE_CONFIDENCE
                it.detection.confidence >= threshold
            }
            .sortedWith(
                compareBy<DetectionWithDistance> { ANNOUNCEMENT_PRIORITY.indexOf(it.detection.className.lowercase()).let { i -> if (i == -1) 99 else i } }
                    .thenByDescending { it.warningLevel }
                    .thenBy { it.distanceInfo.distanceMeters }
                    .thenByDescending { it.detection.confidence }
            )
    }

    private fun announceDetections(detections: List<DetectionWithDistance>, currentTime: Long) {
        val toAnnounce = detections.firstOrNull {
            currentTime - (lastAnnouncedObjects[it.detection.className] ?: 0L) > announceInterval
        }
        toAnnounce?.let {
            vibrateBasedOnDistance(it.distanceInfo.category)
            val message = buildAnnouncementMessage(it)
            speak(message)
            lastAnnouncedObjects[it.detection.className] = currentTime
            Log.i(TAG, "Announced: $message [${it.detection.confidencePercent}%]")
        }
    }

    private fun buildAnnouncementMessage(detWithDist: DetectionWithDistance): String {
        val detection = detWithDist.detection; val distInfo = detWithDist.distanceInfo
        val warningLevel = detWithDist.warningLevel; val objectNameIndo = translateObjectName(detection.className)
        val position = getObjectPosition(detection); val distanceText = formatDistanceText(distInfo.distanceMeters)
        return when {
            warningLevel == DepthEstimator.WarningLevel.CRITICAL -> "Awas! Ada $objectNameIndo sangat dekat, $distanceText di $position."
            warningLevel == DepthEstimator.WarningLevel.HIGH -> "Hati-hati, ada $objectNameIndo di $position, jarak $distanceText."
            detection.className.lowercase() == "stair" -> "Terdapat tangga di $position dalam jarak $distanceText."
            else -> "Terdapat $objectNameIndo di $position dalam jarak $distanceText."
        }
    }

    private fun translateObjectName(className: String): String {
        return when (className.lowercase()) {
            "person" -> "orang"; "chair" -> "kursi"; "table" -> "meja"
            "door" -> "pintu"; "stair", "stairs" -> "tangga"; else -> className
        }
    }

    private fun getObjectPosition(detection: Detection): String {
        val imageWidth = if (currentImageWidth > 0) currentImageWidth else _screenWidth.value
        if (imageWidth <= 0) return "depan"
        val relativePosition = detection.bbox.centerX / imageWidth
        return when {
            relativePosition < 0.35f -> "sebelah kiri"
            relativePosition < 0.65f -> "depan"
            else -> "sebelah kanan"
        }
    }

    private fun vibrateBasedOnDistance(category: DepthEstimator.DistanceCategory) {
        vibrator?.let { vib ->
            if (!vib.hasVibrator()) return
            try {
                val pattern: LongArray; val amplitudes: IntArray?
                when (category) {
                    DepthEstimator.DistanceCategory.VERY_CLOSE -> { pattern = longArrayOf(0, 80, 40, 80, 40, 80); amplitudes = intArrayOf(0, 255, 0, 255, 0, 255) }
                    DepthEstimator.DistanceCategory.CLOSE -> { pattern = longArrayOf(0, 120, 80, 120); amplitudes = intArrayOf(0, 200, 0, 200) }
                    DepthEstimator.DistanceCategory.MEDIUM -> { pattern = longArrayOf(0, 150); amplitudes = intArrayOf(0, 150) }
                    else -> { pattern = longArrayOf(0, 100); amplitudes = intArrayOf(0, 100) }
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    val effect = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && !vib.hasAmplitudeControl()) {
                        VibrationEffect.createWaveform(pattern, -1)
                    } else {
                        VibrationEffect.createWaveform(pattern, amplitudes, -1)
                    }
                    vib.vibrate(effect)
                } else {
                    @Suppress("DEPRECATION") vib.vibrate(pattern, -1)
                }
            } catch (e: Exception) { Log.e(TAG, "Error saat memberikan getaran", e) }
        }
    }

    private fun speak(text: String, flushQueue: Boolean = false) {
        tts?.let { engine ->
            val queueMode = if (flushQueue) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
            engine.speak(text, queueMode, null, "utterance_${System.currentTimeMillis()}")
        }
    }

    override fun onCleared() {
        super.onCleared()
        Log.i(TAG, "Clearing DetectionViewModel resources...")
        try {
            processingJob?.cancel()
            tts?.stop(); tts?.shutdown(); tts = null
            detector.close()
            persistentTracker.reset()
            stairTracker.clear(); otherTracker.clear()
            com.thesis.navigationassistance.ml.ImageUtils.clearPools()
            Log.i(TAG, "DetectionViewModel cleared successfully.")
        } catch (e: Exception) { Log.e(TAG, "Error during ViewModel cleanup", e) }
    }

    private fun Bitmap?.recycleSafely() {
        this?.let { if (!it.isRecycled) it.recycle() }
    }
}