package com.example.gdl_text_recognition.ml

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * 1:1 port of ml/preprocess.py — THE SINGLE SOURCE OF TRUTH IS THE PYTHON FILE.
 * Any change there must be mirrored here and re-verified with GoldenVectorTest.
 * Constant names and function names match the Python originals.
 *
 * OpenCV-Android is the same native C++ as Python's cv2, so cvtColor,
 * GaussianBlur, adaptiveThreshold, connectedComponentsWithStats, resize
 * (INTER_AREA) and copyMakeBorder produce numerically identical output.
 *
 * Pipeline stages (photo -> model-ready crops):
 *   0. normalizeSize      : scale to fixed working width (see note below)
 *   1. toGrayscale        : BGR -> gray
 *   2. binarize           : adaptive threshold -> white ink on black background
 *   3. segmentLines       : horizontal projection profile -> line bands
 *   4. segmentCharacters  : connected components within a line band,
 *                           merged vertically (i/j dots), sorted left-to-right
 *   5. toEmnist           : square-pad crop + margin, resize 28x28, normalize [0,1]
 */
object Preprocess {

    // Tunable constants — mirror ml/preprocess.py exactly
    const val WORK_WIDTH = 1000           // target working width; wider photos are downscaled, smaller pass through
    const val ADAPTIVE_BLOCK_SIZE = 31    // odd; neighborhood for adaptive threshold
    const val ADAPTIVE_C = 15.0           // constant subtracted from local mean
    const val LINE_INK_THRESHOLD = 0.005  // row is "ink" if >0.5% white pixels
    const val MIN_LINE_HEIGHT_PX = 8      // discard line bands thinner than this
    const val MIN_COMPONENT_AREA = 15     // discard specks smaller than this (px)
    const val MIN_COMPONENT_DIM = 3       // discard components thinner than this (px)
    const val CHAR_MARGIN_FRAC = 0.15     // margin added around glyph before resize
    const val EMNIST_SIZE = 28
    const val SPACE_GAP_FACTOR = 1.7      // gap > factor * median gap -> word space

    data class CharBox(val x: Int, val y: Int, val w: Int, val h: Int)

    // Stage 0: resolution normalization
    /**
     * Scale the image down to a fixed working width.
     *
     * Every threshold below is expressed in ABSOLUTE pixels. Without this stage
     * their meaning changes with camera resolution: at ~3500 px the adaptive-
     * threshold window is narrower than a pen stroke, so thick strokes hollow
     * out and fragment into vertical slivers. Never upscales.
     *
     * Returns the input Mat unchanged when it is already narrow enough, so the
     * caller must not assume a new Mat is always allocated.
     */
    fun normalizeSize(img: Mat): Mat {
        val w = img.cols()
        val h = img.rows()
        if (w <= WORK_WIDTH) return img
        val scale = WORK_WIDTH.toDouble() / w
        val dst = Mat()
        Imgproc.resize(
            img, dst,
            Size(WORK_WIDTH.toDouble(), Math.round(h * scale).toDouble()),
            0.0, 0.0, Imgproc.INTER_AREA
        )
        return dst
    }

    // Stage 1-2: grayscale + binarization
    fun toGrayscale(imgBgr: Mat): Mat {
        if (imgBgr.channels() == 1) return imgBgr
        val gray = Mat()
        Imgproc.cvtColor(imgBgr, gray, Imgproc.COLOR_BGR2GRAY)
        return gray
    }

    /**
     * Adaptive (local) threshold -> robust to uneven lighting.
     * THRESH_BINARY_INV: dark ink on light paper becomes WHITE on BLACK,
     * matching EMNIST polarity. Output: 8UC1 {0, 255}.
     */
    fun binarize(gray: Mat): Mat {
        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(3.0, 3.0), 0.0)
        val binary = Mat()
        Imgproc.adaptiveThreshold(
            blurred, binary, 255.0,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY_INV,
            ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
        )
        return binary
    }

    // Stage 3: line segmentation (horizontal projection profile)
    /**
     * Returns [y0, y1) bands containing text lines, top to bottom.
     * A row is "inked" if its white-pixel fraction exceeds LINE_INK_THRESHOLD;
     * consecutive inked rows form a band; thin bands are discarded as noise.
     */
    fun segmentLines(binary: Mat): List<IntArray> {
        val rows = binary.rows()
        val cols = binary.cols().toDouble()
        val inked = BooleanArray(rows)
        val rowBuf = ByteArray(binary.cols())
        for (y in 0 until rows) {
            binary.row(y).get(0, 0, rowBuf)
            var white = 0
            for (b in rowBuf) if (b.toInt() != 0) white++
            inked[y] = (white / cols) > LINE_INK_THRESHOLD
        }
        val bands = ArrayList<IntArray>()
        var start = -1
        for (y in 0 until rows) {
            if (inked[y] && start < 0) {
                start = y
            } else if (!inked[y] && start >= 0) {
                if (y - start >= MIN_LINE_HEIGHT_PX) bands.add(intArrayOf(start, y))
                start = -1
            }
        }
        if (start >= 0 && rows - start >= MIN_LINE_HEIGHT_PX) {
            bands.add(intArrayOf(start, rows))
        }
        return bands
    }

    // Stage 4: character segmentation within a line band
    /**
     * Merge components overlapping horizontally (>=50% of the narrower box):
     * the dot of i/j is a separate component above the stem and must be
     * re-united with it before classification.
     */
    private fun mergeVertical(boxesIn: List<CharBox>): List<CharBox> {
        val boxes = boxesIn.sortedBy { it.x }
        val merged = ArrayList<CharBox>()
        for (b in boxes) {
            if (merged.isNotEmpty()) {
                val m = merged.last()
                val overlap = minOf(m.x + m.w, b.x + b.w) - maxOf(m.x, b.x)
                if (overlap > 0 && overlap >= 0.5 * minOf(m.w, b.w)) {
                    val x0 = minOf(m.x, b.x); val y0 = minOf(m.y, b.y)
                    val x1 = maxOf(m.x + m.w, b.x + b.w); val y1 = maxOf(m.y + m.h, b.y + b.h)
                    merged[merged.size - 1] = CharBox(x0, y0, x1 - x0, y1 - y0)
                    continue
                }
            }
            merged.add(b)
        }
        return merged
    }

    /**
     * Connected components -> filter specks -> merge i/j dots -> sort by x.
     * Input: binary crop of ONE line band. Output: boxes in line coordinates.
     */
    fun segmentCharacters(lineBinary: Mat): List<CharBox> {
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val n = Imgproc.connectedComponentsWithStats(lineBinary, labels, stats, centroids, 8)
        val boxes = ArrayList<CharBox>()
        for (i in 1 until n) {  // 0 is background
            val x = stats.get(i, Imgproc.CC_STAT_LEFT)[0].toInt()
            val y = stats.get(i, Imgproc.CC_STAT_TOP)[0].toInt()
            val w = stats.get(i, Imgproc.CC_STAT_WIDTH)[0].toInt()
            val h = stats.get(i, Imgproc.CC_STAT_HEIGHT)[0].toInt()
            val area = stats.get(i, Imgproc.CC_STAT_AREA)[0].toInt()
            if (area < MIN_COMPONENT_AREA || w < MIN_COMPONENT_DIM || h < MIN_COMPONENT_DIM) continue
            boxes.add(CharBox(x, y, w, h))
        }
        labels.release(); stats.release(); centroids.release()
        return mergeVertical(boxes)
    }

    // Stage 5: crop -> EMNIST-style 28x28 float tensor
    /**
     * Input : binary crop (8UC1 {0,255}) tightly around one glyph.
     * Output: FloatArray of size 28*28 in [0,1], white-on-black, glyph centered
     *         in a square canvas with proportional margin.
     * Steps: pad -> resize -> scale. Row-major (y*28 + x), matching
     * numpy .flatten() so it lines up with the exported golden JSON.
     */
    fun toEmnist(glyphBinary: Mat): FloatArray {
        val h = glyphBinary.rows()
        val w = glyphBinary.cols()
        val side = maxOf(h, w)
        val margin = Math.round(side * CHAR_MARGIN_FRAC).toInt()
        val canvasSide = side + 2 * margin

        val top = (canvasSide - h) / 2
        val bottom = canvasSide - h - top
        val left = (canvasSide - w) / 2
        val right = canvasSide - w - left

        val canvas = Mat()
        Core.copyMakeBorder(
            glyphBinary, canvas, top, bottom, left, right,
            Core.BORDER_CONSTANT, Scalar(0.0)
        )

        val resized = Mat()
        Imgproc.resize(
            canvas, resized,
            Size(EMNIST_SIZE.toDouble(), EMNIST_SIZE.toDouble()),
            0.0, 0.0, Imgproc.INTER_AREA
        )

        val out = FloatArray(EMNIST_SIZE * EMNIST_SIZE)
        val buf = ByteArray(EMNIST_SIZE * EMNIST_SIZE)
        resized.get(0, 0, buf)
        for (i in out.indices) {
            out[i] = (buf[i].toInt() and 0xFF) / 255.0f
        }
        canvas.release(); resized.release()
        return out
    }

    // Word spacing: decide where spaces go from inter-character gaps
    /**
     * flags[i] == true if a SPACE precedes character i (i>=1).
     * Rule: gap between consecutive boxes > SPACE_GAP_FACTOR * median gap.
     */
    fun detectSpaces(boxes: List<CharBox>): List<Boolean> {
        if (boxes.size < 2) return List(boxes.size) { false }
        val gaps = IntArray(boxes.size - 1) { i ->
            boxes[i + 1].x - (boxes[i].x + boxes[i].w)
        }
        val positive = gaps.filter { it > 0 }.sorted()
        if (positive.isEmpty()) return List(boxes.size) { false }
        val medianGap = positive[positive.size / 2]
        val threshold = maxOf(SPACE_GAP_FACTOR * medianGap, 1.0)
        val flags = ArrayList<Boolean>(boxes.size)
        flags.add(false)
        for (g in gaps) flags.add(g > threshold)
        return flags
    }
}