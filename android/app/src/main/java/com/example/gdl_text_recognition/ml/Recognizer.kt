package com.example.gdl_text_recognition.ml

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * 1:1 port of the Python `Recognizer` class in ml/pipeline.py.
 *
 * The int8 TFLite model has a FLOAT32 interface (verified: input/output dtype
 * float32, quant (0.0, 0)), so no manual quantization is needed — we feed the
 * 28*28 FloatArray straight in, exactly like Python's set_tensor().
 *
 * Output is 26 float probabilities (already softmaxed inside the graph). The
 * predicted letter is LETTERS[argmax]; confidence is that probability, taken
 * verbatim — do NOT re-apply softmax, or confidence will diverge from the
 * golden vectors.
 */
class Recognizer(context: Context, modelAsset: String = "model_int8.tflite") {

    companion object {
        const val LETTERS = "abcdefghijklmnopqrstuvwxyz"
        const val EMNIST_SIZE = 28
        const val NUM_CLASSES = 26
    }

    private val interpreter: Interpreter

    init {
        val model = loadModelFile(context, modelAsset)
        interpreter = Interpreter(model)
    }

    /** Result of classifying one glyph. Mirrors Python's (str, float) tuple. */
    data class Prediction(val letter: Char, val confidence: Float)

    /**
     * Runs one 28x28 crop through the model.
     * @param tensor row-major FloatArray of length 28*28 in [0,1], as produced
     *               by Preprocess.toEmnist().
     */
    fun classify(tensor: FloatArray): Prediction {
        require(tensor.size == EMNIST_SIZE * EMNIST_SIZE) {
            "expected ${EMNIST_SIZE * EMNIST_SIZE} values, got ${tensor.size}"
        }

        // Input buffer: [1, 28, 28, 1] float32, row-major — same layout as the numpy tensor[None] in Python.
        val input = ByteBuffer.allocateDirect(4 * tensor.size).order(ByteOrder.nativeOrder())
        for (v in tensor) input.putFloat(v)
        input.rewind()

        val output = Array(1) { FloatArray(NUM_CLASSES) }
        interpreter.run(input, output)

        val probs = output[0]
        var k = 0
        for (i in 1 until NUM_CLASSES) if (probs[i] > probs[k]) k = i
        return Prediction(LETTERS[k], probs[k])
    }

    fun close() = interpreter.close()

    private fun loadModelFile(context: Context, assetName: String): ByteBuffer {
        // Memory-map the .tflite straight from assets. Requires the model to be
        // stored uncompressed (noCompress += "tflite" in build.gradle.kts),
        // otherwise startOffset/declaredLength won't map correctly.
        val afd = context.assets.openFd(assetName)
        afd.createInputStream().channel.use { channel ->
            return channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
    }
}