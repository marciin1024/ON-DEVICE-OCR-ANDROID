package com.example.gdl_text_recognition

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.gdl_text_recognition.ml.Pipeline
import com.example.gdl_text_recognition.ml.Recognizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

// "Scanner terminal" palette
private val BgDeep    = Color(0xFF12141A)
private val Surface   = Color(0xFF1C1F28)
private val Accent    = Color(0xFF4DB6AC)
private val TextMain  = Color(0xFFE8EAED)
private val TextMuted = Color(0xFF8A909C)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (!OpenCVLoader.initLocal()) {
            Log.e("MainActivity", "OpenCV failed to initialize!")
        }
        enableEdgeToEdge()
        setContent {
            ScannerScreen()
        }
    }
}

@Composable
fun ScannerScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()

    val recognizer = remember { Recognizer(context) }

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                    PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted -> hasCameraPermission = granted }

    androidx.compose.runtime.LaunchedEffect(Unit) {
        if (!hasCameraPermission) permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    val imageCapture = remember { ImageCapture.Builder().build() }
    var transcription by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    var hasResult by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(BgDeep)
            .padding(16.dp)
    ) {
        // Preview with reticle frame
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(3f / 4f)
                .clip(RoundedCornerShape(14.dp))
                .background(Color.Black)
        ) {
            if (hasCameraPermission) {
                CameraPreview(
                    modifier = Modifier.fillMaxSize(),
                    imageCapture = imageCapture,
                    lifecycleOwner = lifecycleOwner
                )
                ReticleOverlay(modifier = Modifier.fillMaxSize())
            } else {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text(
                        "Grant camera access\nto scan text.",
                        color = TextMuted,
                        textAlign = TextAlign.Center,
                        fontSize = 15.sp
                    )
                }
            }
        }

        Spacer(Modifier.height(20.dp))

        // Capture button
        CaptureButton(
            busy = busy,
            enabled = hasCameraPermission && !busy,
            onClick = {
                if (busy) return@CaptureButton
                busy = true
                hasResult = false
                transcription = ""
                imageCapture.takePicture(
                    ContextCompat.getMainExecutor(context),
                    object : ImageCapture.OnImageCapturedCallback() {
                        override fun onCaptureSuccess(image: ImageProxy) {
                            val rotation = image.imageInfo.rotationDegrees
                            val rgba = imageProxyToMat(image)
                            image.close()
                            scope.launch {
                                val text = withContext(Dispatchers.Default) {
                                    runPipeline(rgba, rotation, recognizer)
                                }
                                transcription = text
                                hasResult = true
                                busy = false
                            }
                        }
                        override fun onError(exc: ImageCaptureException) {
                            transcription = "Could not take the photo. Try again."
                            hasResult = true
                            busy = false
                            Log.e("Capture", "takePicture error", exc)
                        }
                    }
                )
            }
        )

        Spacer(Modifier.height(20.dp))

        // Result panel
        ResultPanel(
            busy = busy,
            hasResult = hasResult,
            text = transcription,
            onCopy = { copyToClipboard(context, transcription) },
            onShare = { shareText(context, transcription) }
        )
    }
}

@Composable
private fun CaptureButton(busy: Boolean, enabled: Boolean, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        enabled = enabled,
        modifier = Modifier.fillMaxWidth().height(56.dp),
        shape = RoundedCornerShape(12.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = Accent,
            contentColor = BgDeep,
            disabledContainerColor = Surface,
            disabledContentColor = TextMuted
        )
    ) {
        if (busy) {
            CircularProgressIndicator(
                modifier = Modifier.size(20.dp),
                color = TextMuted,
                strokeWidth = 2.dp
            )
            Text("  Reading...", fontWeight = FontWeight.Medium, fontSize = 16.sp)
        } else {
            Text("Take photo", fontWeight = FontWeight.SemiBold, fontSize = 16.sp)
        }
    }
}

@Composable
private fun ResultPanel(
    busy: Boolean,
    hasResult: Boolean,
    text: String,
    onCopy: () -> Unit,
    onShare: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(Surface)
            .padding(16.dp)
    ) {
        Text(
            "Recognized text",
            color = TextMuted,
            fontSize = 13.sp,
            fontWeight = FontWeight.Medium
        )
        Spacer(Modifier.height(10.dp))

        val scroll = rememberScrollState()
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 64.dp, max = 220.dp)
                .verticalScroll(scroll)
        ) {
            when {
                !hasResult && !busy -> Text(
                    "Point at text and take a photo.",
                    color = TextMuted, fontSize = 15.sp
                )
                busy -> Text(
                    "Processing image...",
                    color = TextMuted, fontSize = 15.sp
                )
                text.isBlank() -> Text(
                    "No text recognized. Try a closer, sharper photo.",
                    color = TextMuted, fontSize = 15.sp
                )
                else -> Text(
                    text = text,
                    color = TextMain,
                    fontSize = 17.sp,
                    fontFamily = FontFamily.Monospace,
                    lineHeight = 26.sp
                )
            }
        }

        // Actions appear only when there is a real result
        if (hasResult && text.isNotBlank()) {
            Spacer(Modifier.height(14.dp))
            Row(modifier = Modifier.fillMaxWidth()) {
                OutlinedButton(
                    onClick = onCopy,
                    modifier = Modifier.height(44.dp),
                    shape = RoundedCornerShape(10.dp),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = Accent)
                ) { Text("Copy", fontSize = 14.sp) }

                Spacer(Modifier.width(12.dp))

                OutlinedButton(
                    onClick = onShare,
                    modifier = Modifier.height(44.dp),
                    shape = RoundedCornerShape(10.dp),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = Accent)
                ) { Text("Share", fontSize = 14.sp) }
            }
        }
    }
}

/** Four reticle corners — the "this is a scanner" signature. */
@Composable
private fun ReticleOverlay(modifier: Modifier = Modifier) {
    Canvas(modifier = modifier) {
        val len = size.minDimension * 0.09f
        val m = size.minDimension * 0.06f
        val stroke = 3f
        val c = Accent
        // top left
        drawLine(c, Offset(m, m), Offset(m + len, m), stroke)
        drawLine(c, Offset(m, m), Offset(m, m + len), stroke)
        // top right
        drawLine(c, Offset(size.width - m, m), Offset(size.width - m - len, m), stroke)
        drawLine(c, Offset(size.width - m, m), Offset(size.width - m, m + len), stroke)
        // bottom left
        drawLine(c, Offset(m, size.height - m), Offset(m + len, size.height - m), stroke)
        drawLine(c, Offset(m, size.height - m), Offset(m, size.height - m - len), stroke)
        // bottom right
        drawLine(c, Offset(size.width - m, size.height - m), Offset(size.width - m - len, size.height - m), stroke)
        drawLine(c, Offset(size.width - m, size.height - m), Offset(size.width - m, size.height - m - len), stroke)
    }
}

@Composable
fun CameraPreview(
    modifier: Modifier = Modifier,
    imageCapture: ImageCapture,
    lifecycleOwner: androidx.lifecycle.LifecycleOwner
) {
    androidx.compose.ui.viewinterop.AndroidView(
        modifier = modifier,
        factory = { ctx ->
            val previewView = PreviewView(ctx).apply {
                scaleType = PreviewView.ScaleType.FILL_CENTER
            }
            val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder().build().also {
                    it.surfaceProvider = previewView.surfaceProvider
                }
                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner, cameraSelector, preview, imageCapture
                    )
                } catch (e: Exception) {
                    Log.e("CameraPreview", "Failed to bind camera", e)
                }
            }, ContextCompat.getMainExecutor(ctx))
            previewView
        }
    )
}

// Conversion + pipeline
private fun imageProxyToMat(image: ImageProxy): Mat {
    val bitmap: Bitmap = image.toBitmap()
    val mat = Mat()
    Utils.bitmapToMat(bitmap, mat)
    return mat
}

private fun rotateMat(src: Mat, degrees: Int): Mat {
    if (degrees == 0) return src.clone()
    val dst = Mat()
    when (degrees) {
        90 -> Core.rotate(src, dst, Core.ROTATE_90_CLOCKWISE)
        180 -> Core.rotate(src, dst, Core.ROTATE_180)
        270 -> Core.rotate(src, dst, Core.ROTATE_90_COUNTERCLOCKWISE)
        else -> src.copyTo(dst)
    }
    return dst
}

private fun runPipeline(rgba: Mat, rotation: Int, recognizer: Recognizer): String {
    val rotated = rotateMat(rgba, rotation)
    rgba.release()
    val bgr = Mat()
    Imgproc.cvtColor(rotated, bgr, Imgproc.COLOR_RGBA2BGR)
    rotated.release()
    val page = Pipeline.transcribe(bgr, recognizer)
    val text = page.text
    page.binary.release()
    bgr.release()
    return text
}

// Actions: copy / share
private fun copyToClipboard(context: Context, text: String) {
    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    clipboard.setPrimaryClip(ClipData.newPlainText("transcription", text))
}

private fun shareText(context: Context, text: String) {
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = "text/plain"
        putExtra(Intent.EXTRA_TEXT, text)
    }
    context.startActivity(Intent.createChooser(intent, "Share transcription"))
}