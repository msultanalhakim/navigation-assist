package com.thesis.navigationassistance.ui.components

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.ArrowForward
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.Camera
import androidx.compose.material.icons.filled.VolumeUp
import androidx.compose.material.icons.filled.PhoneAndroid
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties

@Composable
fun WelcomeDialog(
    onDismiss: () -> Unit,
    onStart: () -> Unit
) {
    var currentStep by remember { mutableStateOf(0) }

    val steps = listOf(
        WelcomeStep(
            icon = Icons.Default.Visibility,
            title = "Selamat Datang",
            description = "Asisten Navigasi untuk Tunanetra\n\nAplikasi ini membantu Anda mendeteksi objek di sekitar menggunakan kamera",
            accessibilityDescription = "Selamat datang di aplikasi asisten navigasi untuk tunanetra"
        ),
        WelcomeStep(
            icon = Icons.Default.Camera,
            title = "Deteksi Real-time",
            description = "Mengenali 5 objek:\n• Tangga\n• Pintu\n• Orang\n• Meja\n• Kursi",
            accessibilityDescription = "Aplikasi dapat mendeteksi tangga, pintu, orang, meja, dan kursi secara real-time"
        ),
        WelcomeStep(
            icon = Icons.Default.VolumeUp,
            title = "Panduan Suara",
            description = "Notifikasi audio otomatis saat objek terdeteksi dengan informasi posisi dan jarak",
            accessibilityDescription = "Aplikasi memberikan panduan suara otomatis saat mendeteksi objek"
        ),
        WelcomeStep(
            icon = Icons.Default.PhoneAndroid,
            title = "Getaran Haptik",
            description = "Vibrasi sebagai feedback tambahan untuk setiap deteksi objek",
            accessibilityDescription = "Aplikasi memberikan getaran sebagai umpan balik saat mendeteksi objek"
        )
    )

    Dialog(
        onDismissRequest = { },
        properties = DialogProperties(
            dismissOnBackPress = false,
            dismissOnClickOutside = false,
            usePlatformDefaultWidth = false
        )
    ) {
        Card(
            modifier = Modifier
                .fillMaxWidth(0.92f)
                .semantics {
                    contentDescription = steps[currentStep].accessibilityDescription
                },
            shape = RoundedCornerShape(28.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(32.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Animated Icon
                AnimatedIcon(steps[currentStep].icon)

                Spacer(modifier = Modifier.height(24.dp))

                // Title
                Text(
                    text = steps[currentStep].title,
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                    textAlign = TextAlign.Center
                )

                Spacer(modifier = Modifier.height(16.dp))

                // Description
                Text(
                    text = steps[currentStep].description,
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurface,
                    textAlign = TextAlign.Center,
                    lineHeight = MaterialTheme.typography.bodyLarge.lineHeight.times(1.4f)
                )

                Spacer(modifier = Modifier.height(32.dp))

                // Step Indicator
                StepIndicator(currentStep = currentStep, totalSteps = steps.size)

                Spacer(modifier = Modifier.height(32.dp))

                // Action Buttons
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    if (currentStep > 0) {
                        OutlinedButton(
                            onClick = { currentStep-- },
                            modifier = Modifier
                                .weight(1f)
                                .height(56.dp)
                                .semantics {
                                    contentDescription = "Tombol kembali ke langkah sebelumnya"
                                },
                            shape = RoundedCornerShape(16.dp)
                        ) {
                            Icon(Icons.Default.ArrowBack, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Kembali", style = MaterialTheme.typography.titleMedium)
                        }
                    } else {
                        TextButton(
                            onClick = onDismiss,
                            modifier = Modifier
                                .weight(1f)
                                .height(56.dp)
                                .semantics {
                                    contentDescription = "Tombol lewati tutorial"
                                },
                            shape = RoundedCornerShape(16.dp)
                        ) {
                            Text("Lewati", style = MaterialTheme.typography.titleMedium)
                        }
                    }

                    Button(
                        onClick = {
                            if (currentStep < steps.size - 1) {
                                currentStep++
                            } else {
                                onStart()
                            }
                        },
                        modifier = Modifier
                            .weight(1f)
                            .height(56.dp)
                            .semantics {
                                contentDescription = if (currentStep < steps.size - 1)
                                    "Tombol lanjut ke langkah berikutnya"
                                else
                                    "Tombol mulai menggunakan aplikasi"
                            },
                        shape = RoundedCornerShape(16.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.primary
                        )
                    ) {
                        Text(
                            text = if (currentStep < steps.size - 1) "Lanjut" else "Mulai",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Icon(
                            if (currentStep < steps.size - 1) Icons.Default.ArrowForward else Icons.Default.Check,
                            contentDescription = null
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun AnimatedIcon(icon: ImageVector) {
    val infiniteTransition = rememberInfiniteTransition(label = "icon_animation")

    val scale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 1.1f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "scale"
    )

    Box(
        modifier = Modifier
            .size(120.dp)
            .scale(scale)
            .background(
                brush = Brush.radialGradient(
                    colors = listOf(
                        MaterialTheme.colorScheme.primaryContainer,
                        MaterialTheme.colorScheme.surface
                    )
                ),
                shape = RoundedCornerShape(30.dp)
            ),
        contentAlignment = Alignment.Center
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            modifier = Modifier.size(64.dp),
            tint = MaterialTheme.colorScheme.primary
        )
    }
}

@Composable
private fun StepIndicator(currentStep: Int, totalSteps: Int) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        repeat(totalSteps) { index ->
            Box(
                modifier = Modifier
                    .size(if (index == currentStep) 32.dp else 10.dp)
                    .background(
                        color = if (index == currentStep)
                            MaterialTheme.colorScheme.primary
                        else
                            MaterialTheme.colorScheme.surfaceVariant,
                        shape = RoundedCornerShape(if (index == currentStep) 8.dp else 50.dp)
                    )
                    .semantics {
                        contentDescription = "Langkah ${index + 1} dari $totalSteps"
                    }
            )
        }
    }
}

private data class WelcomeStep(
    val icon: ImageVector,
    val title: String,
    val description: String,
    val accessibilityDescription: String
)