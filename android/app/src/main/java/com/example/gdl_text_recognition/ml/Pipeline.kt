package com.example.gdl_text_recognition.ml

import org.opencv.core.Mat
import org.opencv.core.Rect

/**
 * 1:1 port of ml/pipeline.py — photo -> transcription, end to end.
 *
 * Glues Preprocess stages to the Recognizer, mirroring exactly what Python's
 * transcribe() does per image. Result structures (CharResult / LineResult /
 * PageResult) match the Python dataclasses, including how line.text inserts a
 * space BEFORE a character and page.text joins lines with '\n'.
 */

/** One classified character. Mirrors Python CharResult. */
data class CharResult(
    val box: Preprocess.CharBox,   // full-image coordinates (y offset by band)
    val tensor: FloatArray,        // 28*28 float32 — what the model saw
    val letter: Char,
    val confidence: Float,
    val spaceBefore: Boolean
)

/** One text line. Mirrors Python LineResult. */
data class LineResult(
    val band: IntArray,            // [y0, y1]
    val chars: MutableList<CharResult> = mutableListOf()
) {
    val text: String
        get() = buildString {
            for (c in chars) {
                if (c.spaceBefore) append(' ')
                append(c.letter)
            }
        }
}

/** One page. Mirrors Python PageResult. */
data class PageResult(
    val binary: Mat,
    val lines: MutableList<LineResult> = mutableListOf()
) {
    val text: String
        get() = lines.joinToString("\n") { it.text }
}

object Pipeline {

    /**
     * Full pipeline on one photo. See Preprocess for stage semantics.
     * @param imgBgr decoded image (BGR or gray Mat, 8-bit).
     */
    fun transcribe(imgBgr: Mat, recognizer: Recognizer, timing: LongArray? = null): PageResult {
        val normalized = Preprocess.normalizeSize(imgBgr)
        val gray = Preprocess.toGrayscale(normalized)
        val binary = Preprocess.binarize(gray)
        val page = PageResult(binary = binary)

        for (band in Preprocess.segmentLines(binary)) {
            val y0 = band[0]
            val y1 = band[1]
            // line_bin = binary[y0:y1]  — a view/crop of the band rows
            val lineBin = Mat(binary, Rect(0, y0, binary.cols(), y1 - y0)).clone()

            val boxes = Preprocess.segmentCharacters(lineBin)
            if (boxes.isEmpty()) {
                lineBin.release()
                continue
            }
            val spaces = Preprocess.detectSpaces(boxes)
            val line = LineResult(band = intArrayOf(y0, y1))

            for (i in boxes.indices) {
                val box = boxes[i]
                val sp = spaces[i]
                // glyph = line_bin[box.y : box.y+box.h, box.x : box.x+box.w]
                val glyph = Mat(lineBin, Rect(box.x, box.y, box.w, box.h)).clone()
                val tensor = Preprocess.toEmnist(glyph)
                val t0 = System.nanoTime()
                val pred = recognizer.classify(tensor)
                if (timing != null) timing[0] += System.nanoTime() - t0
                // global_box: x unchanged, y shifted into full-image coordinates
                val globalBox = Preprocess.CharBox(box.x, box.y + y0, box.w, box.h)
                line.chars.add(
                    CharResult(globalBox, tensor, pred.letter, pred.confidence, sp)
                )
                glyph.release()
            }
            page.lines.add(line)
            lineBin.release()
        }
        return page
    }
}