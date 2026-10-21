package com.example.gdl_text_recognition

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.gdl_text_recognition.ml.Pipeline
import com.example.gdl_text_recognition.ml.Preprocess
import com.example.gdl_text_recognition.ml.Recognizer
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * THE GATE for the Kotlin port.
 *
 * For each golden case: load glyph_N.png from assets, push it through the
 * Kotlin Preprocess.toEmnist(), and assert
 *   (a) every tensor value is within atol (=1/255) of the Python tensor
 *       (tensor_N.json, exported from the same run that produced the model), and
 *   (b) Recognizer.classify() predicts the same letter Python did.
 *
 * GREEN  = preprocessing + inference parity is PROVEN, not assumed.
 * RED (a) = a numeric divergence; the message prints the first mismatching
 *           pixel index and both values (usually resize interpolation, blur,
 *           or a byte-sign bug).
 * RED (b) = same tensor, different prediction — a Recognizer/label issue.
 *
 * Runs on the PHYSICAL DEVICE (androidTest): the point is to prove the exact
 * on-device numerics, so it must not run on the host JVM.
 */
@RunWith(AndroidJUnit4::class)
class GoldenVectorTest {

    private lateinit var context: Context
    private lateinit var recognizer: Recognizer
    private var atol: Double = 1.0 / 255.0

    @Before
    fun setUp() {
        assertTrue("OpenCV failed to initialize", OpenCVLoader.initLocal())
        context = InstrumentationRegistry.getInstrumentation().targetContext
        recognizer = Recognizer(context)
        val manifest = JSONObject(readAsset("golden/manifest.json"))
        atol = manifest.getDouble("atol")
    }

    @Test
    fun kotlinToEmnistMatchesPythonGolden() {
        val manifest = JSONObject(readAsset("golden/manifest.json"))
        val cases = manifest.getJSONArray("cases")

        for (ci in 0 until cases.length()) {
            val case = cases.getJSONObject(ci)
            val glyphName = case.getString("glyph")
            val tensorName = case.getString("tensor").replace(".npy", ".json")
            val expectedLetter = case.getString("expected_letter")

            // load the input glyph exactly as saved (binary PNG, 8UC1)
            val glyph = loadGlyphMat("golden/$glyphName")

            // run the Kotlin port
            val tensor = Preprocess.toEmnist(glyph)

            // (a) pixel-wise parity against the Python tensor
            val expected = readTensorJson("golden/$tensorName")
            assertEquals(
                "$glyphName: tensor length mismatch",
                expected.size, tensor.size
            )
            var firstBad = -1
            var maxDiff = 0.0
            for (i in tensor.indices) {
                val d = Math.abs(tensor[i] - expected[i])
                if (d > maxDiff) maxDiff = d.toDouble()
                if (d > atol && firstBad < 0) firstBad = i
            }
            assertTrue(
                "$glyphName: pixel parity FAILED at index $firstBad " +
                        "(kotlin=${if (firstBad >= 0) tensor[firstBad] else 0f}, " +
                        "python=${if (firstBad >= 0) expected[firstBad] else 0f}, " +
                        "atol=$atol, maxDiff=$maxDiff)",
                firstBad < 0
            )

            // (b) prediction parity
            val pred = recognizer.classify(tensor)
            assertEquals(
                "$glyphName: predicted letter differs from Python",
                expectedLetter, pred.letter.toString()
            )
        }
    }

    // helpers
    /** Load a golden glyph PNG as a single-channel 8-bit binary Mat {0,255}. */
    private fun loadGlyphMat(assetPath: String): Mat {
        val bytes = context.assets.open(assetPath).readBytes()
        val bmp = android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        val rgba = Mat()
        Utils.bitmapToMat(bmp, rgba)          // 8UC4
        val gray = Mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
        rgba.release()
        return gray                            // 8UC1, values already {0,255}
    }

    private fun readTensorJson(assetPath: String): FloatArray {
        val arr = JSONArray(readAsset(assetPath))
        return FloatArray(arr.length()) { i -> arr.getDouble(i).toFloat() }
    }

    private fun readAsset(assetPath: String): String =
        context.assets.open(assetPath).bufferedReader().use { it.readText() }
}