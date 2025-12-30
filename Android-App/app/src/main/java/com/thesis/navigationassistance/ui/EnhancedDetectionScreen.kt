package com.thesis.navigationassistance.ui

import com.thesis.navigationassistance.ml.ImageUtils // Pastikan import ini ada
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Paint // Import Paint
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.thesis.navigationassistance.data.DetectionWithDistance
import com.thesis.navigationassistance.ml.ImageUtils.toBitmap
import com.thesis.navigationassistance.ml.DepthEstimator
import com.thesis.navigationassistance.ui.components.WelcomeDialog
import com.thesis.navigationassistance.viewmodel.DetectionViewModel
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.sin // Import sin

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EnhancedDetectionScreen(viewModel: DetectionViewModel = viewModel()) {
    var showWelcome by remember { mutableStateOf(true) }
    var isReady by remember { mutableStateOf(false) }

    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val detectionsWithDistance by viewModel.detectionsWithDistance.collectAsState()
    val fps by viewModel.fps.collectAsState()

    val previewView = remember { PreviewView(context) }

    val cameraExecutor = remember {
        Executors.newSingleThreadExecutor { r ->
            Thread(r, "CameraInference-Thread").apply {
                priority = Thread.MAX_PRIORITY
            }
        }
    }

    var cameraWidth by remember { mutableStateOf(640) }
    var cameraHeight by remember { mutableStateOf(480) }
    var showMenu by remember { mutableStateOf(false) }
    var showAbout by remember { mutableStateOf(false) }

    DisposableEffect(Unit) {
        previewView.post {
            viewModel.setScreenWidth(previewView.width)
        }
        onDispose {
            cameraExecutor.shutdown()
        }
    }

    LaunchedEffect(isReady) {
        if (!isReady) return@LaunchedEffect

        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        try {
                            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                            val baseBitmap = imageProxy.toBitmap() // Konversi tanpa rotasi
                            val rotatedBitmap = if (rotationDegrees != 0) {
                                // Panggil rotasi dari ImageUtils
                                ImageUtils.rotateBitmap(baseBitmap, rotationDegrees.toFloat()).also {
                                    baseBitmap.recycle()
                                }
                            } else {
                                baseBitmap
                            }

                            if (cameraWidth != rotatedBitmap.width || cameraHeight != rotatedBitmap.height) {
                                cameraWidth = rotatedBitmap.width
                                cameraHeight = rotatedBitmap.height
                            }

                            viewModel.processImageAsync(rotatedBitmap) // Kirim bitmap yang sudah benar orientasinya

                        } catch (e: Exception) {
                            Log.e("DetectionScreen", "Frame error", e)
                        } finally {
                            imageProxy.close()
                        }
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner, cameraSelector, preview, imageAnalyzer
                )
                Log.d("DetectionScreen", "Camera bound (Optimized Pipeline)")
            } catch (e: Exception) {
                Log.e("DetectionScreen", "Camera binding failed", e)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    Box(modifier = Modifier.fillMaxSize()) {
        if (showWelcome) {
            WelcomeDialog(
                onDismiss = { showWelcome = false; isReady = true },
                onStart = { showWelcome = false; isReady = true }
            )
        }

        AndroidView(
            factory = { previewView },
            modifier = Modifier
                .fillMaxSize()
                .then(if (showMenu) Modifier.blur(8.dp) else Modifier)
        )

        EnhancedDetectionOverlayWithDistance(
            detectionsWithDistance = detectionsWithDistance,
            cameraWidth = cameraWidth,
            cameraHeight = cameraHeight,
            modifier = Modifier.fillMaxSize()
        )

        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .height(120.dp)
                .align(Alignment.TopCenter),
            color = Color.Transparent
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        brush = Brush.verticalGradient(
                            colors = listOf(Color.Black.copy(alpha = 0.7f), Color.Transparent)
                        )
                    )
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 20.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(
                        modifier = Modifier.semantics { contentDescription = "Aplikasi Asisten Navigasi untuk Tunanetra" }
                    ) {
                        Text(
                            text = "Asisten Navigasi",
                            color = Color.White,
                            fontSize = 24.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "untuk Tunanetra",
                            color = Color.White.copy(alpha = 0.8f),
                            fontSize = 14.sp
                        )
                    }
                    IconButton(
                        onClick = { showMenu = !showMenu },
                        modifier = Modifier.semantics { contentDescription = if (showMenu) "Tutup menu" else "Buka menu" }
                    ) {
                        Icon(
                            if (showMenu) Icons.Default.Close else Icons.Default.Menu,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.size(28.dp)
                        )
                    }
                }
            }
        }

        AnimatedVisibility(
            visible = !showMenu,
            modifier = Modifier.align(Alignment.BottomCenter).padding(16.dp),
            enter = slideInVertically(initialOffsetY = { it }, animationSpec = spring(stiffness = Spring.StiffnessMedium)) + fadeIn(),
            exit = slideOutVertically(targetOffsetY = { it }, animationSpec = spring(stiffness = Spring.StiffnessMedium)) + fadeOut()
        ) {
            ModernStatsCardWithDistance(
                detectionsWithDistance = detectionsWithDistance,
                fps = fps,
                cameraWidth = cameraWidth,
                cameraHeight = cameraHeight
            )
        }

        AnimatedVisibility(
            visible = showMenu,
            modifier = Modifier.align(Alignment.CenterEnd),
            enter = slideInHorizontally(initialOffsetX = { it }) + fadeIn(),
            exit = slideOutHorizontally(targetOffsetX = { it }) + fadeOut()
        ) {
            SideMenu(
                onShowHelp = { showMenu = false; showWelcome = true },
                onShowAbout = { showMenu = false; showAbout = true },
                onClose = { showMenu = false }
            )
        }

        if (showAbout) {
            AboutDialog(onDismiss = { showAbout = false })
        }
    }
}

// Composable ModernStatsCardWithDistance, DetectionItemWithDistance, SideMenu, MenuItem, AboutDialog, translateObjectName

@Composable
private fun ModernStatsCardWithDistance(
    detectionsWithDistance: List<DetectionWithDistance>,
    fps: Float,
    cameraWidth: Int,
    cameraHeight: Int
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = "Panel statistik: ${detectionsWithDistance.size} objek, FPS ${fps.toInt()}"
            },
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.95f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        Icons.Default.Visibility, // Ikon diganti
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Status Deteksi",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                }

                Surface(
                    shape = RoundedCornerShape(12.dp),
                    color = when {
                        fps >= 20f -> Color(0xFF4CAF50)
                        fps >= 12f -> Color(0xFFFFC107)
                        else -> Color(0xFFF44336)
                    }.copy(alpha = 0.2f)
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            Icons.Default.Speed,
                            contentDescription = null,
                            modifier = Modifier.size(16.dp),
                            tint = when {
                                fps >= 20f -> Color(0xFF4CAF50)
                                fps >= 12f -> Color(0xFFFFC107)
                                else -> Color(0xFFF44336)
                            }
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = "${fps.toInt()} FPS",
                            style = MaterialTheme.typography.labelLarge,
                            fontWeight = FontWeight.Bold,
                            color = when {
                                fps >= 20f -> Color(0xFF4CAF50)
                                fps >= 12f -> Color(0xFFFFC107)
                                else -> Color(0xFFF44336)
                            }
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = if (detectionsWithDistance.isEmpty()) "Memindai..."
                else "${detectionsWithDistance.size} Objek Terdeteksi",
                style = MaterialTheme.typography.bodyLarge,
                color = if (detectionsWithDistance.isEmpty())
                    MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                else
                    MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Medium
            )

            if (detectionsWithDistance.isNotEmpty()) {
                Spacer(modifier = Modifier.height(12.dp))
                HorizontalDivider(color = MaterialTheme.colorScheme.surfaceVariant)
                Spacer(modifier = Modifier.height(12.dp))

                detectionsWithDistance
                    .sortedWith(DetectionWithDistance.ByConfidenceDescending)
                    .take(3)
                    .forEachIndexed { idx, detWithDist ->
                        DetectionItemWithDistance(
                            rank = idx + 1,
                            detectionWithDistance = detWithDist
                        )
                        if (idx < 2 && idx < detectionsWithDistance.size - 1) {
                            Spacer(modifier = Modifier.height(8.dp))
                        }
                    }

                if (detectionsWithDistance.size > 3) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "+${detectionsWithDistance.size - 3} objek lainnya",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                    )
                }
            }
        }
    }
}

@Composable
private fun DetectionItemWithDistance(
    rank: Int,
    detectionWithDistance: DetectionWithDistance
) {
    val detection = detectionWithDistance.detection
    val distInfo = detectionWithDistance.distanceInfo
    val warningLevel = detectionWithDistance.warningLevel

    val icon = when (detection.className.lowercase()) {
        "stair" -> Icons.Default.Stairs
        "door" -> Icons.Default.MeetingRoom
        "person" -> Icons.Default.Person
        "table" -> Icons.Default.TableRestaurant
        "chair" -> Icons.Default.Chair
        else -> Icons.Default.HelpOutline
    }

    val colorScheme = when (warningLevel) {
        DepthEstimator.WarningLevel.CRITICAL -> Color(0xFFD32F2F)
        DepthEstimator.WarningLevel.HIGH -> Color(0xFFFF6F00)
        DepthEstimator.WarningLevel.MEDIUM -> Color(0xFFFBC02D)
        else -> when (detection.className.lowercase()) {
            "stair" -> Color(0xFFFF5252)
            "door" -> Color(0xFF448AFF)
            "person" -> Color(0xFF69F0AE)
            "table" -> Color(0xFFE040FB)
            "chair" -> Color(0xFFFDD835)
            else -> Color.Gray
        }
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = "Peringkat $rank: ${translateObjectName(detection.className)}, " +
                        "jarak ${distInfo.indonesianText}, " +
                        "keyakinan ${detection.confidencePercent} persen"
            },
        verticalAlignment = Alignment.CenterVertically
    ) {
        Surface(
            shape = RoundedCornerShape(8.dp),
            color = colorScheme.copy(alpha = 0.15f)
        ) {
            Text(
                text = "#$rank",
                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold,
                color = colorScheme
            )
        }
        Spacer(modifier = Modifier.width(12.dp))
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = colorScheme,
            modifier = Modifier.size(24.dp)
        )
        Spacer(modifier = Modifier.width(12.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = translateObjectName(detection.className),
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium
            )
            Text(
                text = distInfo.indonesianText,
                style = MaterialTheme.typography.bodySmall,
                color = colorScheme,
                fontWeight = FontWeight.Medium
            )
        }
        Column(horizontalAlignment = Alignment.End) {
            Text(
                text = "${detection.confidencePercent}%",
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Bold,
                color = colorScheme
            )
            Box(modifier = Modifier.width(60.dp).height(4.dp).background(colorScheme.copy(alpha = 0.2f), RoundedCornerShape(2.dp))) {
                Box(modifier = Modifier.fillMaxWidth(detection.confidence).fillMaxHeight().background(colorScheme, RoundedCornerShape(2.dp)))
            }
        }
    }
}


@Composable
private fun SideMenu(
    onShowHelp: () -> Unit,
    onShowAbout: () -> Unit,
    onClose: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxHeight()
            .width(280.dp)
            .semantics { contentDescription = "Menu samping aplikasi" },
        shape = RoundedCornerShape(topStart = 24.dp, bottomStart = 24.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 16.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Menu",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold
                )
                IconButton(onClick = onClose, modifier = Modifier.semantics { contentDescription = "Tutup menu" }) {
                    Icon(Icons.Default.Close, contentDescription = null, tint = MaterialTheme.colorScheme.onSurface)
                }
            }
            Spacer(modifier = Modifier.height(32.dp))
            MenuItem(icon = Icons.Default.HelpOutline, title = "Tutorial", description = "Lihat panduan penggunaan", onClick = onShowHelp) // Ikon diganti
            Spacer(modifier = Modifier.height(16.dp))
            MenuItem(
                icon = Icons.Default.Info,
                title = "Tentang Aplikasi",
                description = "Informasi aplikasi",
                onClick = onShowAbout
            )
            Spacer(modifier = Modifier.weight(1f))
            HorizontalDivider()
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = "Versi 1.0.0\n© 2025 Navigation Assistance",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth()
            )
        }
    }
}

@Composable
private fun MenuItem(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    title: String,
    description: String,
    onClick: () -> Unit
) {
    Surface(
        onClick = onClick,
        shape = RoundedCornerShape(16.dp),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
        modifier = Modifier.fillMaxWidth().semantics { contentDescription = "$title: $description" }
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(imageVector = icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(32.dp))
            Spacer(modifier = Modifier.width(16.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(text = title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                Text(text = description, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f))
            }
            Icon(Icons.Default.ChevronRight, contentDescription = null, tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f))
        }
    }
}

@Composable
private fun AboutDialog(onDismiss: () -> Unit) {
    Dialog(onDismissRequest = onDismiss) {
        Card(
            modifier = Modifier.fillMaxWidth().padding(16.dp),
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
        ) {
            Column(
                modifier = Modifier.fillMaxWidth().padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Icon(
                    Icons.Default.Info,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(64.dp)
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text(text = "Asisten Navigasi untuk Tunanetra", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center)
                Spacer(modifier = Modifier.height(8.dp))
                Text(text = "Versi 1.0.0", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f))
                Spacer(modifier = Modifier.height(24.dp))
                HorizontalDivider()
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "Teknologi yang digunakan:\n" +
                            "• Deteksi Objek: YOLOv11\n" +
                            "• Estimasi Jarak: Metode Monokular\n" +
                            "• Pelacakan Objek: Persistent Tracker\n" +
                            "• AI Engine: TensorFlow Lite\n" +
                            "• Output Suara: Text-to-Speech Android",
                    style = MaterialTheme.typography.bodyMedium, textAlign = TextAlign.Start, lineHeight = 20.sp
                )
                Spacer(modifier = Modifier.height(24.dp))
                Text(
                    text = "© 2025 Navigation Assistance\nDirancang untuk membantu mobilitas tunanetra.",
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f), textAlign = TextAlign.Center, lineHeight = 18.sp
                )
                Spacer(modifier = Modifier.height(24.dp))
                Button(onClick = onDismiss, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(12.dp)) {
                    Text("Tutup")
                }
            }
        }
    }
}

private fun translateObjectName(className: String): String {
    return when (className.lowercase()) {
        "person" -> "Orang"
        "chair" -> "Kursi"
        "table" -> "Meja"
        "door" -> "Pintu"
        "stair", "stairs" -> "Tangga"
        else -> className.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
    }
}