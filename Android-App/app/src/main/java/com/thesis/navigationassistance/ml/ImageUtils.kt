package com.thesis.navigationassistance.ml

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.util.Log
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

object ImageUtils {

    private const val TAG = "ImageUtils"

    // Matrix yang dapat digunakan kembali untuk rotasi
    private val rotationMatrix = Matrix()

    // Pool ByteBuffer untuk mengurangi alokasi
    private val byteBufferPool = mutableListOf<ByteBuffer>()
    private const val MAX_BUFFER_POOL_SIZE = 3

    /**
     * Konversi bitmap ke ByteBuffer dengan buffer reuse.
     * Ini dipanggil OLEH YoloDetector.kt yang baru.
     */
    fun bitmapToByteBuffer(
        bitmap: Bitmap,
        inputSize: Int,
        numChannels: Int
    ): ByteBuffer {
        // Fungsi ini seharusnya tidak dipanggil lagi oleh YoloDetector baru,
        // tapi kita biarkan untuk kompatibilitas jika ada pemanggilan lain.
        // YoloDetector baru menggunakan getOrCreateBuffer secara langsung.
        Log.w(TAG, "Peringatan: bitmapToByteBuffer(legacy) dipanggil.")

        val resizedBitmap = if (bitmap.width != inputSize || bitmap.height != inputSize) {
            Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false) // false = lebih cepat
        } else {
            bitmap
        }

        val bufferSize = 4 * inputSize * inputSize * numChannels
        val byteBuffer = getOrCreateBuffer(bufferSize)

        val pixels = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        var pixelIndex = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val pixel = pixels[pixelIndex++]

                // Optimasi bit shifting
                val r = (pixel shr 16 and 0xFF).toFloat() / 255f
                val g = (pixel shr 8 and 0xFF).toFloat() / 255f
                val b = (pixel and 0xFF).toFloat() / 255f

                byteBuffer.putFloat(r)
                byteBuffer.putFloat(g)
                byteBuffer.putFloat(b)
            }
        }

        byteBuffer.rewind()

        if (resizedBitmap != bitmap) {
            resizedBitmap.recycle()
        }

        return byteBuffer
    }

    /**
     * Mengambil atau membuat ByteBuffer dari pool.
     * Digunakan oleh YoloDetector.kt yang baru.
     */
    @Synchronized
    fun getOrCreateBuffer(size: Int): ByteBuffer {
        // Coba cari buffer di pool
        val index = byteBufferPool.indexOfFirst { it.capacity() >= size }

        return if (index >= 0) {
            val buffer = byteBufferPool.removeAt(index)
            buffer.clear()
            buffer
        } else {
            ByteBuffer.allocateDirect(size).apply {
                order(ByteOrder.nativeOrder())
            }
        }
    }

    /**
     * Mengembalikan buffer ke pool untuk digunakan kembali.
     * Digunakan oleh YoloDetector.kt yang baru.
     */
    @Synchronized
    fun recycleBuffer(buffer: ByteBuffer) {
        if (byteBufferPool.size < MAX_BUFFER_POOL_SIZE) {
            buffer.clear()
            byteBufferPool.add(buffer)
        }
    }

    /**
     * Optimalkan bitmap dengan kualitas adaptif
     */
    fun optimizeBitmapForProcessing(
        bitmap: Bitmap,
        maxDimension: Int = 1920,
        maintainAspectRatio: Boolean = true
    ): Bitmap {
        val width = bitmap.width
        val height = bitmap.height

        if (width <= maxDimension && height <= maxDimension) {
            return bitmap
        }

        val scale = minOf(
            maxDimension.toFloat() / width,
            maxDimension.toFloat() / height
        )

        val newWidth = (width * scale).toInt()
        val newHeight = (height * scale).toInt()

        return try {
            Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, false) // false = lebih cepat
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OOM during optimization")
            bitmap
        }
    }

    /**
     * Konversi Android Image ke Bitmap - VERSI EFISIEN
     * Menghapus rotasi otomatis (ensurePortraitOrientation)
     */
    fun Image.toBitmap(): Bitmap {
        val startTime = System.currentTimeMillis()

        Log.d(TAG, "Image.toBitmap() START")
        Log.d(TAG, "   Camera Image: ${width}x${height}")
        Log.d(TAG, "   Format: ${formatToString(format)}")

        // STEP 1: Konversi berdasarkan format
        val rawBitmap = when (format) {
            ImageFormat.JPEG -> {
                Log.d(TAG, "Processing JPEG")
                decodeJPEGOptimized()
            }
            ImageFormat.YUV_420_888 -> {
                Log.d(TAG, "Processing YUV_420_888")
                yuv420ToBitmapOptimized()
            }
            else -> {
                // EnhancedDetectionScreen meminta RGBA_8888, jadi ini path utamanya
                Log.d(TAG, "Processing RGBA")
                rgbaToBitmapOptimized()
            }
        }

        if (rawBitmap == null) {
            Log.e(TAG, "Failed to decode image!")
            throw IllegalStateException("Failed to decode image")
        }

        Log.d(TAG, "Raw bitmap: ${rawBitmap.width}x${rawBitmap.height}")

        // STEP 2: [DIHAPUS] ensurePortraitOrientation(rawBitmap) dihapus.
        // Rotasi sekarang ditangani oleh EnhancedDetectionScreen.

        // STEP 3: Optimalkan jika terlalu besar (opsional, tapi bagus)
        val optimizedBitmap = if (rawBitmap.width > 1280 || rawBitmap.height > 1280) {
            Log.d(TAG, "Optimizing large bitmap...")
            val optimized = optimizeBitmapForProcessing(rawBitmap, 1280)
            if (optimized != rawBitmap) {
                rawBitmap.recycle()
            }
            optimized
        } else {
            rawBitmap
        }

        val totalTime = System.currentTimeMillis() - startTime
        Log.d(TAG, "Processing time: ${totalTime}ms")
        Log.d(TAG, "Final: ${optimizedBitmap.width}x${optimizedBitmap.height} (Un-rotated)")

        return optimizedBitmap
    }

    /**
     * Decode JPEG dengan options
     */
    private fun Image.decodeJPEGOptimized(): Bitmap? {
        val buffer = planes[0].buffer
        val bytes = ByteArray(buffer.capacity())
        buffer.get(bytes)

        val options = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inMutable = false // Immutable = lebih cepat
        }

        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options)
    }

    /**
     * YUV to Bitmap dengan konversi efisien
     */
    private fun Image.yuv420ToBitmapOptimized(): Bitmap? {
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride

        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        val rgbBytes = IntArray(width * height)

        for (y in 0 until height) {
            val yLineStart = y * yRowStride
            val uvLineStart = (y / 2) * uvRowStride

            for (x in 0 until width) {
                val yValue = (yBuffer[yLineStart + x * yPixelStride].toInt() and 0xFF)

                val uvOffset = uvLineStart + (x / 2) * uvPixelStride
                val uValue = (uBuffer[uvOffset].toInt() and 0xFF) - 128
                val vValue = (vBuffer[uvOffset].toInt() and 0xFF) - 128

                // YUV->RGB conversion
                val r = (yValue + 1.370705f * vValue).toInt().coerceIn(0, 255)
                val g = (yValue - 0.337633f * uValue - 0.698001f * vValue).toInt().coerceIn(0, 255)
                val b = (yValue + 1.732446f * uValue).toInt().coerceIn(0, 255)

                rgbBytes[y * width + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(rgbBytes, 0, width, 0, 0, width, height)

        return bitmap
    }


    /**
     * RGBA to Bitmap (direct copy)
     */
    private fun Image.rgbaToBitmapOptimized(): Bitmap {
        val buffer = planes[0].buffer
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        buffer.rewind()
        bitmap.copyPixelsFromBuffer(buffer)
        return bitmap
    }

    // [DIHAPUS] Fungsi ensurePortraitOrientation dan getOrientationName dihapus
    // karena rotasi kini ditangani di UI layer (EnhancedDetectionScreen).

    /**
     * Helper: Format enum to string
     */
    private fun formatToString(format: Int): String {
        return when (format) {
            ImageFormat.JPEG -> "JPEG"
            ImageFormat.YUV_420_888 -> "YUV_420_888"
            ImageFormat.NV21 -> "NV21"
            else -> "RGBA_8888 (format=$format)"
        }
    }

    /**
     * Rotasi bitmap dengan matrix reuse & fast mode
     * Fungsi ini dipanggil oleh EnhancedDetectionScreen
     */
    fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        return rotateBitmapOptimized(bitmap, degrees)
    }

    /**
     * Rotasi cepat dengan reusable matrix
     */
    private fun rotateBitmapOptimized(bitmap: Bitmap, degrees: Float): Bitmap {
        if (degrees == 0f) return bitmap

        // Reuse matrix object
        synchronized(rotationMatrix) {
            rotationMatrix.reset()
            rotationMatrix.postRotate(degrees)

            return try {
                Bitmap.createBitmap(
                    bitmap, 0, 0,
                    bitmap.width, bitmap.height,
                    rotationMatrix,
                    false // false = lebih cepat, tanpa filtering
                )
            } catch (e: OutOfMemoryError) {
                Log.e(TAG, "OOM during rotation")
                bitmap
            }
        }
    }

    /**
     * Flip bitmap dengan matrix reuse
     */
    fun flipBitmapHorizontally(bitmap: Bitmap): Bitmap {
        val matrix = Matrix().apply {
            preScale(-1f, 1f)
        }

        return try {
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, false)
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OOM during flip")
            bitmap
        }
    }

    /**
     * Crop bitmap (fast mode)
     */
    fun cropBitmap(
        bitmap: Bitmap,
        x: Int,
        y: Int,
        width: Int,
        height: Int
    ): Bitmap {
        return try {
            Bitmap.createBitmap(bitmap, x, y, width, height)
        } catch (e: Exception) {
            Log.e(TAG, "Crop failed", e)
            bitmap
        }
    }

    /**
     * Scale bitmap with fast mode
     */
    fun scaleBitmap(
        bitmap: Bitmap,
        newWidth: Int,
        newHeight: Int,
        highQuality: Boolean = false // Default false untuk speed
    ): Bitmap {
        return try {
            Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, highQuality)
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OOM during scale")
            bitmap
        }
    }

    /**
     * Get bitmap memory size
     */
    fun Bitmap.getMemorySize(): Int {
        return rowBytes * height
    }

    /**
     * Safe recycle
     */
    fun Bitmap.recycleSafely() {
        if (!isRecycled) {
            recycle()
        }
    }

    /**
     * Validate bitmap for processing
     */
    fun Bitmap.validateForProcessing(): Boolean {
        val isValid = width > 0 && height > 0 && !isRecycled
        if (!isValid) {
            Log.e(TAG, "Invalid bitmap: ${width}x${height}, recycled: $isRecycled")
        }
        return isValid
    }

    /**
     * Print debug info
     */
    fun Bitmap.printDebugInfo(label: String = "Bitmap") {
        Log.d(TAG, "$label Info:")
        Log.d(TAG, "   Size: ${width}x${height}")
        // Log.d(TAG, "   Orientation: ${getOrientationName(this)}") // Dihapus
        Log.d(TAG, "   Config: $config")
        Log.d(TAG, "   Memory: ${getMemorySize() / 1024}KB")
        Log.d(TAG, "   Aspect: ${String.format("%.3f", width.toFloat() / height)}")
    }

    /**
     * Batch recycle multiple bitmaps
     */
    fun recycleBitmaps(vararg bitmaps: Bitmap?) {
        bitmaps.forEach { bitmap ->
            bitmap?.recycleSafely()
        }
    }

    /**
     * Clear all pools (call on cleanup)
     */
    @Synchronized
    fun clearPools() {
        byteBufferPool.clear()
        Log.d(TAG, "Cleared buffer pools")
    }

    /**
     * Get pool statistics
     */
    @Synchronized
    fun getPoolStats(): String {
        return "ByteBuffer pool: ${byteBufferPool.size}/$MAX_BUFFER_POOL_SIZE"
    }
}