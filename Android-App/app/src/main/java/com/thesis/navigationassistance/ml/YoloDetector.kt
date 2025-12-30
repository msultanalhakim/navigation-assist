package com.thesis.navigationassistance.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.util.Log
import com.thesis.navigationassistance.data.BoundingBox
import com.thesis.navigationassistance.data.Detection
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer // Pastikan import ini ada
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min


class YoloDetector(
    private val context: Context,
    private val modelPath: String = "yolov11.tflite",
    private val labelPath: String = "labels.txt",
    private val useGpu: Boolean = true
) {
    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()
    private var gpuDelegate: GpuDelegate? = null
    private var accelerationEnabled = false
    private var accelerationType = "CPU"
    private var letterboxBitmapCache: Bitmap? = null
    private val matrix = Matrix()
    private val reusableLetterboxInfo = LetterboxInfo()

    companion object {
        private const val TAG = "YoloDetector"
        // Kembalikan ke 640 sesuai ekspektasi model
        const val INPUT_SIZE = 640
        private const val NUM_CHANNELS = 3
        private const val OUTPUT_SIZE = 8400 // Biarkan 8400
        private const val NUM_OUTPUTS = 9
        private const val NUM_CLASSES = 5
        private const val MAX_DETECTIONS_PER_CLASS = 15
        private const val ABSOLUTE_MAX_DETECTIONS = 50
    }

    private data class ClassValidationRules(
        val minWidth: Float,
        val minHeight: Float,
        val maxAreaRatio: Float,
        val aspectRatioRange: ClosedFloatingPointRange<Float>,
        val aspectExclusionRange: ClosedFloatingPointRange<Float>? = null,
        val allowEdgeDetection: Boolean,
        val edgeConfidenceThreshold: Float,
        val description: String
    )

    private val validationRules = mapOf(
        "door" to ClassValidationRules(
            minWidth = 12f, minHeight = 25f, maxAreaRatio = 0.90f,
            aspectRatioRange = 0.20f..3.5f, aspectExclusionRange = null,
            allowEdgeDetection = true, edgeConfidenceThreshold = 0.58f,
            description = "Door: stricter validation"
        ),
        "person" to ClassValidationRules(
            minWidth = 15f, minHeight = 30f, maxAreaRatio = 0.85f,
            aspectRatioRange = 0.18f..2.0f, aspectExclusionRange = null,
            allowEdgeDetection = true, edgeConfidenceThreshold = 0.60f,
            description = "Person: stricter validation"
        ),
        "stair" to ClassValidationRules(
            minWidth = 35f, minHeight = 25f, maxAreaRatio = 0.90f,
            aspectRatioRange = 0.4f..3.5f, aspectExclusionRange = null,
            allowEdgeDetection = true, edgeConfidenceThreshold = 0.62f,
            description = "Stair: stricter validation"
        ),
        "chair" to ClassValidationRules(
            minWidth = 12f, minHeight = 20f, maxAreaRatio = 0.75f,
            aspectRatioRange = 0.35f..1.8f, aspectExclusionRange = 0.9f..1.1f, // Blokir kipas
            allowEdgeDetection = true, edgeConfidenceThreshold = 0.65f,
            description = "Chair: Strict, with circular exclusion"
        ),
        "table" to ClassValidationRules(
            minWidth = 25f, minHeight = 18f, maxAreaRatio = 0.80f,
            aspectRatioRange = 0.25f..5.0f, aspectExclusionRange = null,
            allowEdgeDetection = true, edgeConfidenceThreshold = 0.63f,
            description = "Table: stricter validation"
        )
    )

    init {
        loadLabels()
        loadModel()
        warmupGPU()
    }

    private fun loadLabels() {
        try {
            val reader = BufferedReader(InputStreamReader(context.assets.open(labelPath)))
            labels = reader.readLines()
            reader.close()
            Log.d(TAG, "Loaded ${labels.size} labels: $labels")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading labels", e)
            labels = listOf("person", "chair", "table", "door", "stair")
        }
    }

    private fun loadModel() {
        try {
            val options = Interpreter.Options()

            if (useGpu && initializeGpuDelegate(options)) {
                accelerationEnabled = true
                accelerationType = "GPU"
                Log.d(TAG, "GPU Acceleration ENABLED")
            } else {
                options.setNumThreads(4)
                options.setUseXNNPACK(true)
                accelerationType = "CPU (XNNPACK)"
                Log.d(TAG, "CPU with XNNPACK optimization")
            }

            val model = loadModelFile(modelPath)
            interpreter = Interpreter(model, options)

            Log.d(TAG, "Model loaded with $accelerationType")
            // Log ini sekarang akan mencetak 640 lagi
            Log.d(TAG, "  Input: [1,$INPUT_SIZE,$INPUT_SIZE,3] | Output: [1,$NUM_OUTPUTS,$OUTPUT_SIZE]")

        } catch (e: Exception) {
            Log.e(TAG, "Error loading model", e)
            cleanupDelegates()
        }
    }

    private fun initializeGpuDelegate(options: Interpreter.Options): Boolean {
        return try {
            val compatibilityList = CompatibilityList()
            if (compatibilityList.isDelegateSupportedOnThisDevice) {
                val delegateOptions = compatibilityList.bestOptionsForThisDevice
                gpuDelegate = GpuDelegate(delegateOptions)
                options.addDelegate(gpuDelegate)
                Log.d(TAG, "GPU Delegate created successfully")
                true
            } else {
                Log.w(TAG, "GPU not compatible on this device")
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "GPU Delegate initialization failed", e)
            false
        }
    }

    private fun warmupGPU() {
        if (!accelerationEnabled) return
        try {
            Log.d(TAG, "Warming up GPU...")
            val dummyBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
            repeat(3) {
                detect(dummyBitmap, confidenceThreshold = 0.9f, iouThreshold = 0.5f)
            }
            dummyBitmap.recycle()
            Log.d(TAG, "GPU warmed up")
        } catch (e: Exception) {
            Log.w(TAG, "GPU warmup failed", e)
        }
    }

    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    data class LetterboxInfo(
        var scale: Float = 0f,
        var padLeft: Float = 0f,
        var padTop: Float = 0f,
        var originalWidth: Int = 0,
        var originalHeight: Int = 0,
        var scaledWidth: Int = 0,
        var scaledHeight: Int = 0,
        var orientation: String = ""
    ) {
        fun update(s: Float, pl: Float, pt: Float, ow: Int, oh: Int, sw: Int, sh: Int) {
            scale = s; padLeft = pl; padTop = pt; originalWidth = ow; originalHeight = oh;
            scaledWidth = sw; scaledHeight = sh;
            orientation = if (oh > ow * 1.2) "PORTRAIT" else if (ow > oh * 1.2) "LANDSCAPE" else "SQUARE"
        }
    }

    private fun bitmapToTfliteBuffer(
        bitmap: Bitmap,
        outInfo: LetterboxInfo
    ): ByteBuffer {
        val scale = minOf(INPUT_SIZE.toFloat() / bitmap.width, INPUT_SIZE.toFloat() / bitmap.height)
        val scaledWidth = (bitmap.width * scale).toInt()
        val scaledHeight = (bitmap.height * scale).toInt()
        val padLeft = (INPUT_SIZE - scaledWidth) / 2f
        val padTop = (INPUT_SIZE - scaledHeight) / 2f
        outInfo.update(scale, padLeft, padTop, bitmap.width, bitmap.height, scaledWidth, scaledHeight)

        matrix.reset()
        matrix.postScale(scale, scale)
        matrix.postTranslate(padLeft, padTop)

        // Ukuran buffer sekarang akan dihitung berdasarkan INPUT_SIZE = 640
        val bufferSize = 4 * INPUT_SIZE * INPUT_SIZE * NUM_CHANNELS
        val byteBuffer = ImageUtils.getOrCreateBuffer(bufferSize)

        val letterboxBitmap = if (letterboxBitmapCache?.width == INPUT_SIZE &&
            letterboxBitmapCache?.height == INPUT_SIZE &&
            letterboxBitmapCache?.isRecycled == false) {
            letterboxBitmapCache!!
        } else {
            letterboxBitmapCache?.recycle()
            Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888).also {
                letterboxBitmapCache = it
            }
        }
        val canvas = Canvas(letterboxBitmap)
        canvas.drawColor(Color.rgb(114, 114, 114))
        canvas.drawBitmap(bitmap, matrix, null)

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        letterboxBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        var pixelIndex = 0
        for (i in 0 until INPUT_SIZE) {
            for (j in 0 until INPUT_SIZE) {
                val pixel = pixels[pixelIndex++]
                byteBuffer.putFloat(((pixel shr 16 and 0xFF) / 255f)) // R
                byteBuffer.putFloat(((pixel shr 8 and 0xFF) / 255f))  // G
                byteBuffer.putFloat(((pixel and 0xFF) / 255f))       // B
            }
        }
        byteBuffer.rewind()
        return byteBuffer
    }

    private fun applyNMSOptimized(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        if (detections.isEmpty()) return emptyList()
        val perClassCap = MAX_DETECTIONS_PER_CLASS
        val absoluteCap = ABSOLUTE_MAX_DETECTIONS
        val keptAll = mutableListOf<Detection>()
        val byClass = detections.groupBy { it.classId }
        for ((_, group) in byClass) {
            val sorted = group.sortedByDescending { it.confidence }
            val selected = mutableListOf<Detection>()
            for (det in sorted) {
                var suppress = false
                for (sel in selected) {
                    if (calculateIoUFast(sel.bbox, det.bbox) > iouThreshold) {
                        suppress = true; break
                    }
                }
                if (!suppress) {
                    selected.add(det)
                    if (selected.size >= perClassCap) break
                }
            }
            keptAll.addAll(selected)
        }
        return keptAll.sortedByDescending { it.confidence }.take(absoluteCap)
    }

    fun detect(bitmap: Bitmap, confidenceThreshold: Float = 0.60f, iouThreshold: Float = 0.50f): List<Detection> {
        val (detections, _) = detectInternal(bitmap, confidenceThreshold, iouThreshold, false)
        return detections
    }

    data class DebugInfo(
        val processingTimeMs: Long,
        val rawDetectionsCount: Int,
        val finalDetectionsCount: Int,
        val topPredictions: List<Triple<String, Float, String>>,
        val letterboxInfo: LetterboxInfo,
        val validationFailures: List<String>
    )

    fun detectWithDebug(bitmap: Bitmap, confidenceThreshold: Float = 0.60f, iouThreshold: Float = 0.50f): Pair<List<Detection>, DebugInfo> {
        val (detections, debugInfo) = detectInternal(bitmap, confidenceThreshold, iouThreshold, true)
        return Pair(detections, debugInfo!!)
    }

    private fun detectInternal(bitmap: Bitmap, confidenceThreshold: Float, iouThreshold: Float, isDebug: Boolean): Pair<List<Detection>, DebugInfo?> {
        val startTime = System.currentTimeMillis()
        val inputBuffer = bitmapToTfliteBuffer(bitmap, reusableLetterboxInfo)
        val info = reusableLetterboxInfo
        val outputBuffer = Array(1) { Array(NUM_OUTPUTS) { FloatArray(OUTPUT_SIZE) } }
        try {
            interpreter?.run(inputBuffer, outputBuffer)
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e) // Error akan terjadi di sini
            val errorDebugInfo = if (isDebug) DebugInfo(0, 0, 0, emptyList(), info, listOf("Inference failed: ${e.message}")) else null
            return Pair(emptyList(), errorDebugInfo)
        } finally {
            ImageUtils.recycleBuffer(inputBuffer)
        }
        val topPredictions = if (isDebug) collectTopPredictions(outputBuffer[0]) else emptyList()
        if (isDebug) logTopPredictions(topPredictions)
        val (rawDetections, validationFailures) = postProcessOptimized(outputBuffer[0], info, confidenceThreshold, isDebug)
        val finalDetections = applyNMSOptimized(rawDetections, iouThreshold)
        val totalTime = System.currentTimeMillis() - startTime
        if (!isDebug) Log.d(TAG, "Inference: ${totalTime}ms [$accelerationType] | ${rawDetections.size} raw -> ${finalDetections.size} final")
        val debugInfo = if (isDebug) DebugInfo(totalTime, rawDetections.size, finalDetections.size, topPredictions.take(10), info, validationFailures) else null
        return Pair(finalDetections, debugInfo)
    }

    private fun collectTopPredictions(output: Array<FloatArray>): List<Triple<String, Float, String>> {
        val topPredictions = mutableListOf<Triple<String, Float, String>>()
        val iterLimit = minOf(200, OUTPUT_SIZE)
        for (i in 0 until iterLimit) {
            for (c in 0 until NUM_CLASSES) {
                val prob = output[4 + c][i].coerceIn(0f, 1f)
                if (prob > 0.4f) {
                    val className = labels.getOrElse(c) { "unknown" }
                    val xCenter = output[0][i]; val yCenter = output[1][i]
                    topPredictions.add(Triple(className, prob, "Anchor $i at (${xCenter.toInt()}, ${yCenter.toInt()})"))
                }
            }
        }
        return topPredictions.sortedByDescending { it.second }
    }

    private fun logTopPredictions(topPredictions: List<Triple<String, Float, String>>) {
        Log.d(TAG, "================ DEBUG: TOP 10 RAW PREDICTIONS ================")
        topPredictions.take(10).forEachIndexed { idx, (cls, prob, reason) -> Log.d(TAG, "   ${idx+1}. $cls: ${(prob*100).toInt()}% | $reason") }
        Log.d(TAG, "==============================================================")
    }

    private fun postProcessOptimized(output: Array<FloatArray>, info: LetterboxInfo, confidenceThreshold: Float, isDebug: Boolean): Pair<List<Detection>, List<String>> {
        val detections = mutableListOf<Detection>(); val detectionsPerClass = mutableMapOf<Int, Int>(); val validationFailureLog = mutableListOf<String>()
        val scaledWidth = info.scaledWidth; val scaledHeight = info.scaledHeight
        val validLeft = info.padLeft; val validTop = info.padTop; val validRight = info.padLeft + scaledWidth; val validBottom = info.padTop + scaledHeight
        var totalProcessed = 0; var validGeometry = 0; var insideImage = 0; var passedConfidence = 0; var passedValidation = 0
        val allRawDetections = if (isDebug) mutableListOf<Pair<String, Pair<Float, BoundingBox>>>() else null

        for (i in 0 until OUTPUT_SIZE) {
            totalProcessed++
            if (detections.size >= ABSOLUTE_MAX_DETECTIONS) { if (isDebug) Log.d(TAG, "Early termination: max detections"); break }
            val xCenterNorm = output[0][i]; val yCenterNorm = output[1][i]; val widthNorm = output[2][i]; val heightNorm = output[3][i]
            if (!isValidGeometry(xCenterNorm, yCenterNorm, widthNorm, heightNorm)) continue; validGeometry++
            // Koordinat sekarang dalam skala 640
            val xCenterScaled = xCenterNorm * INPUT_SIZE; val yCenterScaled = yCenterNorm * INPUT_SIZE
            val widthScaled = widthNorm * INPUT_SIZE; val heightScaled = heightNorm * INPUT_SIZE
            val x1_scaled_pre = xCenterScaled - widthScaled / 2f; val y1_scaled_pre = yCenterScaled - heightScaled / 2f
            val x2_scaled_pre = xCenterScaled + widthScaled / 2f; val y2_scaled_pre = yCenterScaled + heightScaled / 2f
            if (!isCenterInBoundsRelaxed(xCenterScaled, yCenterScaled, validLeft, validTop, validRight, validBottom)) continue; insideImage++
            var maxProb = 0f; var maxClassId = -1
            for (c in 0 until NUM_CLASSES) { val prob = output[4 + c][i].coerceIn(0f, 1f); if (prob > maxProb) { maxProb = prob; maxClassId = c } }
            if (maxProb < confidenceThreshold) continue; passedConfidence++
            val classCount = detectionsPerClass.getOrDefault(maxClassId, 0); if (classCount >= MAX_DETECTIONS_PER_CLASS) continue
            val className = labels.getOrElse(maxClassId) { "unknown" }
            val overlapRatio = calculateOverlapRatio(x1_scaled_pre, y1_scaled_pre, x2_scaled_pre, y2_scaled_pre, validLeft, validTop, validRight, validBottom, widthScaled * heightScaled)
            val coverage = (widthScaled * heightScaled) / (INPUT_SIZE * INPUT_SIZE); val minOverlap = if (coverage > 0.6f) 0.4f else if (coverage > 0.4f) 0.5f else 0.55f
            if (overlapRatio < minOverlap) continue
            val x1_img = max(0f, x1_scaled_pre - info.padLeft); val y1_img = max(0f, y1_scaled_pre - info.padTop)
            val x2_img = min(scaledWidth.toFloat(), x2_scaled_pre - info.padLeft); val y2_img = min(scaledHeight.toFloat(), y2_scaled_pre - info.padTop)
            if (info.scale <= 0f) continue
            val x1 = (x1_img / info.scale).coerceIn(0f, info.originalWidth.toFloat()); val y1 = (y1_img / info.scale).coerceIn(0f, info.originalHeight.toFloat())
            val x2 = (x2_img / info.scale).coerceIn(0f, info.originalWidth.toFloat()); val y2 = (y2_img / info.scale).coerceIn(0f, info.originalHeight.toFloat())
            val bboxWidth = x2 - x1; val bboxHeight = y2 - y1
            if (isDebug) allRawDetections?.add(Pair(className, Pair(maxProb, BoundingBox(x1, y1, x2, y2))))
            val rules = validationRules[className.lowercase()] ?: ClassValidationRules(15f, 20f, 0.75f, 0.15f..4.5f, null, true, 0.55f, "default")
            if (!validateDetectionEnhanced(bboxWidth, bboxHeight, className, info, x1, y1, x2, y2, maxProb, rules, isDebug, validationFailureLog)) continue; passedValidation++
            detections.add(Detection(BoundingBox(x1, y1, x2, y2), maxClassId, className, maxProb)); detectionsPerClass[maxClassId] = classCount + 1
        }
        if (isDebug && allRawDetections != null && allRawDetections.isNotEmpty()) {
            Log.d(TAG, "================ DEBUG: RAW DETECTIONS (Before Validation) ================")
            allRawDetections.sortedByDescending { it.second.first }.take(10).forEachIndexed { idx, (cls, pair) ->
                val (conf, bbox) = pair; val ar = if (bbox.height > 0) bbox.width / bbox.height else 0f
                Log.d(TAG, "   ${idx+1}. $cls: ${(conf*100).toInt()}% | ${bbox.width.toInt()}x${bbox.height.toInt()} | AR: ${"%.2f".format(ar)}")
            }
            Log.d(TAG, "==========================================================================")
        }
        Log.d(TAG, "PostProcess Stats: $totalProcessed -> $validGeometry -> $insideImage -> $passedConfidence -> $passedValidation -> ${detections.size}")
        return Pair(detections, validationFailureLog)
    }

    private fun validateDetectionEnhanced(width: Float, height: Float, className: String, info: LetterboxInfo, x1: Float, y1: Float, x2: Float, y2: Float, confidence: Float, rules: ClassValidationRules, isDebug: Boolean, failureLog: MutableList<String>): Boolean {
        if (width <= 0f || height <= 0f) { if (isDebug) failureLog.add("$className: Zero dimensions"); return false }
        val aspectRatio = if (height > 0f) width / height else 0f
        if (width < rules.minWidth || height < rules.minHeight) { if (isDebug) failureLog.add("$className: Size too small (${width.toInt()}x${height.toInt()})"); return false }
        val bboxArea = width * height; val imageArea = info.originalWidth * info.originalHeight.toFloat(); if (imageArea <= 0f) return false
        if (bboxArea / imageArea > rules.maxAreaRatio) { if (isDebug) failureLog.add("$className: Area ratio exceeded (${"%.2f".format(bboxArea/imageArea)} > ${rules.maxAreaRatio})"); return false }
        if (aspectRatio !in rules.aspectRatioRange) { if (isDebug) failureLog.add("$className: Invalid AR ${"%.2f".format(aspectRatio)} (expected ${rules.aspectRatioRange})"); return false }
        rules.aspectExclusionRange?.let { if (aspectRatio in it) { if (isDebug) failureLog.add("$className: Rejected AR ${"%.2f".format(aspectRatio)} (in exclusion zone $it)"); return false } }
        val edgeValid = validateEdgeDetectionRelaxed(x1, y1, x2, y2, info, rules, confidence)
        if (!edgeValid && isDebug) failureLog.add("$className: Failed edge validation (Conf: ${"%.2f".format(confidence)} < ${rules.edgeConfidenceThreshold})")
        return edgeValid
    }

    private fun isValidGeometry(xCenterNorm: Float, yCenterNorm: Float, widthNorm: Float, heightNorm: Float): Boolean {
        return widthNorm > 0f && heightNorm > 0f && xCenterNorm in -0.15f..1.15f && yCenterNorm in -0.15f..1.15f
    }

    private fun isCenterInBoundsRelaxed(xCenter: Float, yCenter: Float, left: Float, top: Float, right: Float, bottom: Float): Boolean {
        val margin = 45f
        return xCenter >= (left - margin) && xCenter <= (right + margin) && yCenter >= (top - margin) && yCenter <= (bottom + margin)
    }

    private fun calculateOverlapRatio(x1: Float, y1: Float, x2: Float, y2: Float, validLeft: Float, validTop: Float, validRight: Float, validBottom: Float, bboxArea: Float): Float {
        val ox1 = max(x1, validLeft); val oy1 = max(y1, validTop); val ox2 = min(x2, validRight); val oy2 = min(y2, validBottom)
        val overlapW = max(0f, ox2 - ox1); val overlapH = max(0f, oy2 - oy1); val overlapArea = overlapW * overlapH
        return if (bboxArea > 0) overlapArea / bboxArea else 0f
    }

    private fun validateEdgeDetectionRelaxed(x1: Float, y1: Float, x2: Float, y2: Float, info: LetterboxInfo, rules: ClassValidationRules, confidence: Float): Boolean {
        val margin = 10f
        val onEdge = x1 < margin || y1 < margin || x2 > (info.originalWidth - margin) || y2 > (info.originalHeight - margin)
        if (!onEdge) return true
        val threshold = if (rules.allowEdgeDetection) rules.edgeConfidenceThreshold - 0.03f else rules.edgeConfidenceThreshold
        return confidence >= threshold
    }

    private fun calculateIoUFast(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = max(box1.x1, box2.x1); val y1 = max(box1.y1, box2.y1); val x2 = min(box1.x2, box2.x2); val y2 = min(box1.y2, box2.y2)
        val intersectionW = max(0f, x2 - x1); val intersectionH = max(0f, y2 - y1)
        if (intersectionW == 0f || intersectionH == 0f) return 0f
        val intersection = intersectionW * intersectionH
        val union = box1.area + box2.area - intersection
        return if (union > 0f) intersection / union else 0f
    }

    fun getAccelerationInfo(): String {
        return "Acceleration: $accelerationType${if(accelerationEnabled)" ACTIVE" else ""}\nDevice: ${android.os.Build.MODEL}\nAndroid: ${android.os.Build.VERSION.RELEASE}"
    }

    private fun cleanupDelegates() {
        try { gpuDelegate?.close(); gpuDelegate = null }
        catch (e: Exception) { Log.e(TAG, "Error cleaning up delegates", e) }
    }

    fun close() {
        cleanupDelegates(); interpreter?.close(); interpreter = null
        letterboxBitmapCache?.recycle(); letterboxBitmapCache = null
        Log.d(TAG, "YoloDetector closed")
    }
}