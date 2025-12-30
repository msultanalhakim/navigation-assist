package com.thesis.navigationassistance.ml

import android.util.Log
import com.thesis.navigationassistance.data.BoundingBox
import com.thesis.navigationassistance.data.Detection
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.math.max // Pastikan maxOf/minOf/max/min diimpor
import kotlin.math.min

/**
 * PersistentObjectTracker - Pelacakan Lanjutan dari Jauh ke Sangat Dekat
 * Versi Kurang Strict untuk menghindari False Negative
 */
class PersistentObjectTracker {

    companion object {
        private const val TAG = "PersistentTracker"

        private const val MIN_DETECTIONS_FOR_STABILITY = 3
        private const val MAX_MISS_FRAMES_BEFORE_LOST = 12 // Dari 10
        private const val MAX_MISS_FRAMES_FOR_ANNOUNCE = 10 // Dari 8

        // Ambang batas cakupan tetap sama
        private const val COVERAGE_FAR = 0.15f
        private const val COVERAGE_APPROACHING = 0.35f
        private const val COVERAGE_CLOSE = 0.55f
        private const val COVERAGE_VERY_CLOSE = 0.70f

        private const val GROWTH_THRESHOLD = 1.18f // Sedikit lebih sensitif (dari 1.20)

        // [PERUBAHAN 2] Turunkan ambang confidence untuk pengumuman
        private const val MIN_CONFIDENCE_ACCUMULATED_FOR_ANNOUNCE = 0.58f // Dari 0.60
        private const val CONFIDENCE_DECAY_PER_MISS = 0.92f
        private const val SUSPICIOUS_CONFIDENCE_DROP_THRESHOLD = 0.45f
        private const val MIN_CONFIDENCE_TO_KEEP_TRACKING = 0.35f
    }

    enum class DistanceStage {
        FAR, APPROACHING, CLOSE, VERY_CLOSE, LOST
    }

    private data class TrackedObject(
        val id: String,
        val className: String,
        var lastBbox: BoundingBox,
        var lastSeen: Long,
        var framesSeen: Int = 1,
        var framesLost: Int = 0,
        var maxCoverage: Float = 0f,
        var isApproaching: Boolean = false,
        var hasBeenAnnounced: Boolean = false,
        var confidenceHistory: MutableList<Float> = mutableListOf(),
        var coverageHistory: MutableList<Float> = mutableListOf(),
        var velocityX: Float = 0f,
        var velocityY: Float = 0f,
        var predictedX: Float = 0f,
        var predictedY: Float = 0f,
        var accelerationFactor: Float = 1.0f,
        var distanceStage: DistanceStage = DistanceStage.FAR,
        var accumulatedConfidence: Float = 0f,
        var stageFrames: Int = 0
    )

    private val trackedObjects = mutableMapOf<String, TrackedObject>()
    private val recentlyLost = mutableMapOf<String, TrackedObject>()

    data class TrackingResult(
        val continuedTracking: List<Detection>,
        val lostTracking: List<LostObjectInfo>
    )

    data class LostObjectInfo(
        val className: String,
        val lastKnownBbox: BoundingBox,
        val framesSeen: Int,
        val maxCoverage: Float,
        val wasApproaching: Boolean,
        val shouldAnnounce: Boolean,
        val estimatedDistance: Float,
        val distanceStage: DistanceStage
    )

    fun trackFrame(
        detections: List<Detection>,
        currentTime: Long,
        imageWidth: Int,
        imageHeight: Int
    ): TrackingResult {
        if (imageWidth <= 0 || imageHeight <= 0) return TrackingResult(emptyList(), emptyList())
        val imageArea = imageWidth * imageHeight.toFloat()
        if (imageArea <= 0f) return TrackingResult(emptyList(), emptyList())

        val continuedTracking = mutableListOf<Detection>()
        val lostObjectsToAnnounce = mutableListOf<LostObjectInfo>()

        Log.d(TAG, "Tracking Frame: ${detections.size} detections, ${trackedObjects.size} active tracks.")

        updatePredictions(currentTime)

        val matchedTrackIds = mutableSetOf<String>()
        for (detection in detections) {
            if (!detection.isValid()) continue
            val coverage = detection.bbox.area / imageArea
            if (coverage <= 0f) continue

            val bestMatch = findBestMatchWithPrediction(detection, trackedObjects, imageWidth, imageHeight, coverage)

            if (bestMatch != null) {
                updateTrackEnhanced(bestMatch, detection, coverage, currentTime)
                matchedTrackIds.add(bestMatch.id)
                continuedTracking.add(detection)
                Log.d(TAG, "   Matched ${detection.className} (ID: ...${bestMatch.id.takeLast(4)}) - Frames: ${bestMatch.framesSeen}, Stage: ${bestMatch.distanceStage}")
            } else {
                val newTrack = createNewTrack(detection, coverage, currentTime)
                trackedObjects[newTrack.id] = newTrack
                continuedTracking.add(detection)
                Log.d(TAG, "   New Track: ${detection.className} (ID: ...${newTrack.id.takeLast(4)})")
            }
        }

        val unmatchedTracks = trackedObjects.filterKeys { it !in matchedTrackIds }
        for ((id, track) in unmatchedTracks) {
            track.framesLost++
            track.lastSeen = currentTime
            track.accumulatedConfidence *= CONFIDENCE_DECAY_PER_MISS

            Log.d(TAG, "   Track Lost: ${track.className} (ID: ...${id.takeLast(4)}) - Lost: ${track.framesLost}/${MAX_MISS_FRAMES_BEFORE_LOST}, Conf: ${(track.accumulatedConfidence * 100).toInt()}%")

            if (shouldAnnounceAsVeryClose(track)) {
                Log.i(TAG, "   !!! Announcing VERY CLOSE: ${track.className} (ID: ...${id.takeLast(4)})")
                val lostInfo = LostObjectInfo(
                    className = track.className,
                    lastKnownBbox = track.lastBbox,
                    framesSeen = track.framesSeen,
                    maxCoverage = track.maxCoverage,
                    wasApproaching = track.isApproaching,
                    shouldAnnounce = !track.hasBeenAnnounced,
                    estimatedDistance = estimateVeryCloseDistance(track.maxCoverage),
                    distanceStage = track.distanceStage
                )
                if (lostInfo.shouldAnnounce) {
                    lostObjectsToAnnounce.add(lostInfo)
                    track.hasBeenAnnounced = true
                }
                recentlyLost[id] = track
                trackedObjects.remove(id)
            } else if (track.framesLost > MAX_MISS_FRAMES_BEFORE_LOST ||
                (track.accumulatedConfidence < MIN_CONFIDENCE_TO_KEEP_TRACKING && track.framesSeen > 10)) {
                Log.d(TAG, "   Removing Track: ${track.className} (ID: ...${id.takeLast(4)}) - Reason: ${if (track.framesLost > MAX_MISS_FRAMES_BEFORE_LOST) "Timeout" else "Low Confidence"}")
                trackedObjects.remove(id)
            }
        }

        val cleanupTime = currentTime - 3000
        recentlyLost.entries.removeIf { (_, track) -> track.lastSeen < cleanupTime }

        Log.d(TAG, "Tracking Result: ${continuedTracking.size} continued, ${lostObjectsToAnnounce.size} announced as very close.")
        return TrackingResult(continuedTracking, lostObjectsToAnnounce)
    }

    private fun updatePredictions(currentTime: Long) {
        trackedObjects.values.forEach { track ->
            val timeDelta = (currentTime - track.lastSeen) / 1000f
            if (timeDelta > 0f && timeDelta < 0.1f) {
                track.predictedX = track.lastBbox.centerX + track.velocityX * timeDelta * track.accelerationFactor
                track.predictedY = track.lastBbox.centerY + track.velocityY * timeDelta * track.accelerationFactor
            } else {
                track.predictedX = track.lastBbox.centerX
                track.predictedY = track.lastBbox.centerY
            }
        }
    }

    private fun findBestMatchWithPrediction(
        detection: Detection,
        tracks: Map<String, TrackedObject>,
        imageWidth: Int,
        imageHeight: Int,
        coverage: Float
    ): TrackedObject? {
        var bestMatch: TrackedObject? = null
        var bestScore = -1f

        for ((_, track) in tracks) {
            if (track.className != detection.className) continue
            if (track.framesLost > 5) continue
            if (track.accumulatedConfidence < MIN_CONFIDENCE_TO_KEEP_TRACKING &&
                track.framesSeen > 15 &&
                track.distanceStage != DistanceStage.VERY_CLOSE &&
                track.distanceStage != DistanceStage.CLOSE) {
                Log.w(TAG, "   Skipping low confidence track: ${track.className} (${(track.accumulatedConfidence*100).toInt()}%)")
                continue
            }
            val score = calculateMatchScoreWithPrediction(detection.bbox, track, imageWidth, imageHeight, coverage)
            if (score > bestScore) {
                bestScore = score
                bestMatch = track
            }
        }

        // [PERUBAHAN 3] Turunkan threshold skor sedikit untuk jarak dekat
        val minScoreThreshold = when (bestMatch?.distanceStage) {
            DistanceStage.VERY_CLOSE -> 0.22f // Dari 0.25f
            DistanceStage.CLOSE -> 0.28f // Dari 0.30f
            DistanceStage.APPROACHING -> 0.35f // Dari 0.35f (tetap)
            else -> 0.40f // Dari 0.40f (tetap)
        }

        return if (bestScore > minScoreThreshold) bestMatch else null
    }

    private fun calculateMatchScoreWithPrediction(
        currentBbox: BoundingBox,
        track: TrackedObject,
        imageWidth: Int,
        imageHeight: Int,
        coverage: Float
    ): Float {
        if (imageWidth <= 0 || imageHeight <= 0) return 0f
        val dx = (currentBbox.centerX - track.predictedX) / imageWidth
        val dy = (currentBbox.centerY - track.predictedY) / imageHeight
        val predictedDistanceNormalized = sqrt(dx * dx + dy * dy)
        val maxAllowedDistance = when {
            coverage > COVERAGE_CLOSE -> 0.35f
            coverage > COVERAGE_APPROACHING -> 0.30f
            else -> 0.25f
        }
        if (predictedDistanceNormalized > maxAllowedDistance) return 0f
        val distanceScore = 1f - (predictedDistanceNormalized / maxAllowedDistance)

        val currentArea = currentBbox.area
        val trackedArea = track.lastBbox.area
        if (trackedArea <= 0f || currentArea <= 0f) return 0f
        val areaRatio = min(currentArea, trackedArea) / max(currentArea, trackedArea)
        val minAllowedAreaRatio = if (track.isApproaching) 0.25f else 0.33f
        val sizeScore = if (areaRatio >= minAllowedAreaRatio) areaRatio else 0f

        val iou = currentBbox.iou(track.lastBbox)
        val confidenceScore = track.accumulatedConfidence.coerceIn(0f, 1f)

        return distanceScore * 0.40f + sizeScore * 0.25f + iou * 0.25f + confidenceScore * 0.10f
    }

    private fun createNewTrack(
        detection: Detection,
        coverage: Float,
        currentTime: Long
    ): TrackedObject {
        val id = "${detection.className}_${System.nanoTime()}"
        val initialStage = determineDistanceStage(coverage)
        return TrackedObject(
            id = id,
            className = detection.className,
            lastBbox = detection.bbox,
            lastSeen = currentTime,
            maxCoverage = coverage,
            confidenceHistory = mutableListOf(detection.confidence),
            coverageHistory = mutableListOf(coverage),
            distanceStage = initialStage,
            accumulatedConfidence = detection.confidence,
            predictedX = detection.bbox.centerX,
            predictedY = detection.bbox.centerY
        )
    }

    private fun updateTrackEnhanced(
        track: TrackedObject,
        detection: Detection,
        coverage: Float,
        currentTime: Long
    ) {
        val timeDelta = (currentTime - track.lastSeen) / 1000f
        if (timeDelta > 0.001f && timeDelta < 0.2f) {
            val newVelX = (detection.bbox.centerX - track.lastBbox.centerX) / timeDelta
            val newVelY = (detection.bbox.centerY - track.lastBbox.centerY) / timeDelta
            track.velocityX = track.velocityX * 0.7f + newVelX * 0.3f
            track.velocityY = track.velocityY * 0.7f + newVelY * 0.3f
        }

        val previousArea = track.lastBbox.area
        val currentArea = detection.bbox.area
        if (previousArea > 0f) {
            val growthRatio = currentArea / previousArea
            if (growthRatio > GROWTH_THRESHOLD && coverage > COVERAGE_FAR) {
                track.isApproaching = true
                track.accelerationFactor = 1.2f
            } else if (growthRatio < 0.95f) {
                track.isApproaching = false
                track.accelerationFactor = 1.0f
            }
        }

        val previousStage = track.distanceStage
        val newStage = determineDistanceStage(coverage)
        if (newStage != previousStage) {
            Log.d(TAG, "      Stage Change ${track.className}: $previousStage -> $newStage (ID: ...${track.id.takeLast(4)})")
            track.stageFrames = 1
        } else {
            track.stageFrames++
        }
        track.distanceStage = newStage

        track.lastBbox = detection.bbox
        track.lastSeen = currentTime
        track.framesSeen++
        track.framesLost = 0
        track.maxCoverage = max(track.maxCoverage, coverage)

        val previousAvgConf = track.confidenceHistory.average().toFloat().takeIf { !it.isNaN() } ?: track.accumulatedConfidence
        val confidenceDroppedSignificantly = detection.confidence < SUSPICIOUS_CONFIDENCE_DROP_THRESHOLD
        if (track.distanceStage != DistanceStage.VERY_CLOSE &&
            track.framesSeen > 8 &&
            previousAvgConf > 0.65f &&
            confidenceDroppedSignificantly) {
            Log.w(TAG, "      Suspicious Confidence Drop: ${track.className} from ~${(previousAvgConf*100).toInt()}% to ${(detection.confidence*100).toInt()}%")
            track.accumulatedConfidence = track.accumulatedConfidence * 0.5f + detection.confidence * 0.5f
        } else {
            track.accumulatedConfidence = track.accumulatedConfidence * 0.8f + detection.confidence * 0.2f
        }
        track.accumulatedConfidence = track.accumulatedConfidence.coerceIn(0f, 1f)

        track.confidenceHistory.add(detection.confidence)
        if (track.confidenceHistory.size > 10) track.confidenceHistory.removeAt(0)
        track.coverageHistory.add(coverage)
        if (track.coverageHistory.size > 10) track.coverageHistory.removeAt(0)
    }

    private fun determineDistanceStage(coverage: Float): DistanceStage {
        return when {
            coverage > COVERAGE_VERY_CLOSE -> DistanceStage.VERY_CLOSE
            coverage > COVERAGE_CLOSE -> DistanceStage.CLOSE
            coverage > COVERAGE_APPROACHING -> DistanceStage.APPROACHING
            else -> DistanceStage.FAR
        }
    }

    private fun shouldAnnounceAsVeryClose(track: TrackedObject): Boolean {
        if (track.framesSeen < MIN_DETECTIONS_FOR_STABILITY) return false
        if (track.distanceStage != DistanceStage.CLOSE && track.distanceStage != DistanceStage.VERY_CLOSE) return false
        if (track.maxCoverage < COVERAGE_CLOSE) return false
        // [PERUBAHAN 4] Longgarkan syarat 'isApproaching'. Boleh jika pernah approaching.
        // if (!track.isApproaching) return false -> Hapus syarat ini
        if (track.hasBeenAnnounced) return false
        if (track.accumulatedConfidence < MIN_CONFIDENCE_ACCUMULATED_FOR_ANNOUNCE) return false

        // [PERUBAHAN 5] Longgarkan cek tren cakupan. Cukup tidak menurun drastis.
        if (track.coverageHistory.size >= 3) {
            val recent = track.coverageHistory.takeLast(3)
            // Cek apakah frame terakhir tidak jauh lebih kecil dari frame sebelumnya
            val notDecreasingSignificantly = recent[2] >= recent[1] * 0.95f // Boleh stabil atau turun sedikit
            if (!notDecreasingSignificantly) {
                Log.d(TAG,"      Not announcing ${track.className}: Coverage significantly decreased (${recent.joinToString { "%.2f".format(it) }})")
                return false
            }
        } else {
            return false // Belum cukup data tren
        }

        // Gunakan batas frame hilang yang baru
        if (track.framesLost > MAX_MISS_FRAMES_FOR_ANNOUNCE) return false

        Log.d(TAG, "      Criteria Met for Announce: ${track.className}")
        Log.d(TAG, "         Frames Seen: ${track.framesSeen}, Stage: ${track.distanceStage}, Max Coverage: ${(track.maxCoverage*100).toInt()}%")
        Log.d(TAG, "         Confidence: ${(track.accumulatedConfidence*100).toInt()}%")
        return true
    }

    private fun estimateVeryCloseDistance(maxCoverage: Float): Float {
        return when {
            maxCoverage > 0.90f -> 0.3f
            maxCoverage > 0.80f -> 0.5f
            maxCoverage > 0.70f -> 0.7f
            maxCoverage > COVERAGE_CLOSE -> 0.8f // Cakupan antara 0.55 - 0.70
            else -> 1.0f // Jika hilang saat masih di tahap CLOSE
        }
    }

    fun getTrackingStatus(): String {
        return buildString {
            appendLine("Active Tracks (${trackedObjects.size}):")
            trackedObjects.forEach { (_, track) -> appendLine("  - ${track.className} (ID: ...${track.id.takeLast(4)}): ${track.framesSeen}f, Stage=${track.distanceStage}, Cov=${(track.maxCoverage*100).toInt()}%, Conf=${(track.accumulatedConfidence*100).toInt()}%") }
            appendLine("Recently Lost (${recentlyLost.size}):")
            recentlyLost.forEach { (_, track) -> appendLine("  - ${track.className} (ID: ...${track.id.takeLast(4)}): LastSeen=${track.lastSeen}") }
        }
    }

    fun reset() {
        trackedObjects.clear()
        recentlyLost.clear()
        Log.i(TAG, "Tracking reset.")
    }
}