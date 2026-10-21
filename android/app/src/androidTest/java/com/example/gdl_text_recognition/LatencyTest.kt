package com.example.gdl_text_recognition

import android.content.Context
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.gdl_text_recognition.ml.Pipeline
import com.example.gdl_text_recognition.ml.Recognizer
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * Phase 8 — on-device latency measurement.
 *
 * Runs the full transcribe() pipeline on a fixed, high-resolution image from
 * assets (print_sans_fox.jpg — the same photo that scores CER 0.000 offline),
 * with warm-up, and reports the breakdown:
 *   - total transcribe() time  (CV pipeline + all classifications)
 *   - model inference time      (sum over all classify() calls)
 *   - CV pipeline time          (total - model)
 *
 * Measured from transcribe() only: frame acquisition (ImageProxy -> Bitmap ->
 * Mat) and rotation are NOT included (they add a few ms in the live app).
 * Runs on the physical device.
 */
@RunWith(AndroidJUnit4::class)
class LatencyTest {

    private lateinit var context: Context
    private lateinit var recognizer: Recognizer
    private lateinit var imgBgr: Mat

    companion object {
        const val ASSET = "print_sans_fox.jpg"
        const val WARMUP = 5
        const val RUNS = 20
    }

    @Before
    fun setUp() {
        assertTrue("OpenCV failed to initialize", OpenCVLoader.initLocal())
        context = InstrumentationRegistry.getInstrumentation().targetContext
        recognizer = Recognizer(context)
        imgBgr = loadBgrFromAsset(ASSET)
        Log.d("Latency", "input image: ${imgBgr.cols()} x ${imgBgr.rows()} px")
    }

    @Test
    fun measureTranscribeLatency() {
        // Warm-up: JIT, allocations, interpreter ready — not counted
        repeat(WARMUP) {
            val page = Pipeline.transcribe(imgBgr, recognizer)
            page.binary.release()
        }

        val totalMs = DoubleArray(RUNS)
        val modelMs = DoubleArray(RUNS)
        var charCount = 0

        for (i in 0 until RUNS) {
            val timing = LongArray(1)                 // timing[0] = sum of classify() nanos
            val t0 = System.nanoTime()
            val page = Pipeline.transcribe(imgBgr, recognizer, timing)
            val elapsed = System.nanoTime() - t0

            totalMs[i] = elapsed / 1_000_000.0
            modelMs[i] = timing[0] / 1_000_000.0
            charCount = page.lines.sumOf { it.chars.size }
            page.binary.release()
        }

        val total = stats(totalMs)
        val model = stats(modelMs)
        val cvMs = DoubleArray(RUNS) { totalMs[it] - modelMs[it] }
        val cv = stats(cvMs)

        Log.i("Latency", "==== on-device latency (n=$RUNS, warmup=$WARMUP) ====")
        Log.i("Latency", "characters per page: $charCount")
        Log.i("Latency", "TOTAL transcribe : median=${f(total.median)} mean=${f(total.mean)} " +
                "min=${f(total.min)} max=${f(total.max)} ms")
        Log.i("Latency", "  model (classify): median=${f(model.median)} mean=${f(model.mean)} ms " +
                "(${pct(model.mean, total.mean)}% of total)")
        Log.i("Latency", "  CV pipeline     : median=${f(cv.median)} mean=${f(cv.mean)} ms " +
                "(${pct(cv.mean, total.mean)}% of total)")
        Log.i("Latency", "per-character model time: ${f(model.mean / charCount)} ms/char")
        Log.i("Latency", "=======================================================")

        assertTrue("transcribe produced no characters", charCount > 0)
    }

    // helpers

    private data class Stat(val median: Double, val mean: Double, val min: Double, val max: Double)

    private fun stats(a: DoubleArray): Stat {
        val s = a.sorted()
        val median = if (s.size % 2 == 0) (s[s.size/2 - 1] + s[s.size/2]) / 2 else s[s.size/2]
        return Stat(median, a.average(), s.first(), s.last())
    }

    private fun f(x: Double) = String.format("%.1f", x)
    private fun pct(part: Double, whole: Double) = String.format("%.0f", 100.0 * part / whole)

    private fun loadBgrFromAsset(name: String): Mat {
        val bytes = context.assets.open(name).readBytes()
        val bmp = android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        val rgba = Mat()
        Utils.bitmapToMat(bmp, rgba)                  // 8UC4 RGBA
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release()
        return bgr
    }
}
