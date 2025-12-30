package com.thesis.navigationassistance

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.thesis.navigationassistance.ui.EnhancedDetectionScreen
import com.thesis.navigationassistance.ui.theme.NavigationAssistanceTheme

class MainActivity : ComponentActivity() {

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.all { it.value }
        if (allGranted) {
            startCamera()
        } else {
            Toast.makeText(
                this,
                "Izin kamera dan vibrate diperlukan untuk aplikasi ini",
                Toast.LENGTH_LONG
            ).show()
            finish()
        }
    }

    companion object {
        private const val TAG = "MainActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.d(TAG, "╔═════════════════════════════════════════════════╗")
        Log.d(TAG, "🚀 MainActivity onCreate")
        Log.d(TAG, "   Device: ${android.os.Build.MODEL}")
        Log.d(TAG, "   Android: ${android.os.Build.VERSION.RELEASE} (SDK ${android.os.Build.VERSION.SDK_INT})")
        Log.d(TAG, "   Orientation: ALL supported (Portrait & Landscape)")
        Log.d(TAG, "╚═════════════════════════════════════════════════╝")

        // ✅ OPTIMASI: Hardware acceleration untuk smooth rendering
        window.setFlags(
            WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED,
            WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED
        )

        // Keep screen on
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // ✅ OPTIMASI: Disable window animations untuk faster transitions
        window.attributes = window.attributes.apply {
            windowAnimations = 0
        }

        // Setup fullscreen
        setupFullscreen()

        // ✅ Check permissions
        val permissions = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.VIBRATE
        )

        val allPermissionsGranted = permissions.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }

        if (allPermissionsGranted) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(permissions)
        }
    }

    /**
     * ✅ OPTIMIZED: Fullscreen setup with proper flags
     */
    private fun setupFullscreen() {
        WindowCompat.setDecorFitsSystemWindows(window, false)

        val controller = WindowInsetsControllerCompat(window, window.decorView)
        controller.hide(WindowInsetsCompat.Type.systemBars())
        controller.systemBarsBehavior =
            WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE

        // ✅ OPTIMASI: Set window to max brightness untuk better camera visibility
        window.attributes = window.attributes.apply {
            screenBrightness = WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE
        }
    }

    /**
     * Start camera and compose UI
     */
    private fun startCamera() {
        setContent {
            NavigationAssistanceTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    EnhancedDetectionScreen()
                }
            }
        }

        Log.d(TAG, "✅ Camera started, UI composed")
    }

    override fun onResume() {
        super.onResume()
        setupFullscreen()

        // ✅ OPTIMASI: Log memory status
        logMemoryStatus()
    }

    override fun onPause() {
        super.onPause()
        Log.d(TAG, "⏸️ Activity paused")
    }

    override fun onStop() {
        super.onStop()
        Log.d(TAG, "⏹️ Activity stopped")
    }

    override fun onDestroy() {
        super.onDestroy()

        // ✅ OPTIMASI: Suggest GC on destroy (optional, system will decide)
        System.gc()

        Log.d(TAG, "🗑️ Activity destroyed")
    }

    /**
     * ✅ NEW: Log memory status untuk monitoring
     */
    private fun logMemoryStatus() {
        val runtime = Runtime.getRuntime()
        val usedMemory = (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024
        val maxMemory = runtime.maxMemory() / 1024 / 1024
        val availableMemory = maxMemory - usedMemory

        Log.d(TAG, "📊 Memory Status:")
        Log.d(TAG, "   Used: ${usedMemory}MB / ${maxMemory}MB")
        Log.d(TAG, "   Available: ${availableMemory}MB")

        if (availableMemory < 50) {
            Log.w(TAG, "⚠️ Low memory: ${availableMemory}MB remaining")
        }
    }

    override fun onLowMemory() {
        super.onLowMemory()
        Log.w(TAG, "⚠️ LOW MEMORY WARNING")

        // ✅ OPTIMASI: Force GC pada low memory
        System.gc()
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)

        when (level) {
            TRIM_MEMORY_RUNNING_CRITICAL -> {
                Log.w(TAG, "🚨 TRIM_MEMORY_RUNNING_CRITICAL")
                // Clear caches jika perlu
            }
            TRIM_MEMORY_RUNNING_LOW -> {
                Log.w(TAG, "⚠️ TRIM_MEMORY_RUNNING_LOW")
            }
            TRIM_MEMORY_RUNNING_MODERATE -> {
                Log.d(TAG, "ℹ️ TRIM_MEMORY_RUNNING_MODERATE")
            }
        }
    }
}