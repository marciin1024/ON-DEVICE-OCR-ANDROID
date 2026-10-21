"""
Phase 4 — preprocess.py — SINGLE SOURCE OF TRUTH for image preprocessing.

EVERY transformation here must be ported 1:1 to Android (Recognizer.kt).
Golden-test vectors exported by 04_pipeline.ipynb verify the Kotlin port.
Do not change anything here without regenerating the golden vectors and
re-verifying the port.

Pipeline stages (photo -> model-ready crops):
  1. to_grayscale       : BGR -> gray
  2. binarize           : adaptive threshold -> white ink on black background
                          (matches EMNIST polarity)
  3. segment_lines      : horizontal projection profile -> line bands
  4. segment_characters : connected components within a line band,
                          merged vertically (i/j dots), sorted left-to-right
  5. to_emnist          : square-pad crop + margin, resize 28x28, blur,
                          normalize [0,1]  -> exactly what the CNN expects

Design constraints:
  * only ops with direct OpenCV-Android equivalents (cvtColor, GaussianBlur,
    adaptiveThreshold, connectedComponentsWithStats, resize, copyMakeBorder)
  * no Python-only tricks; everything expressible in Kotlin with the same
    numeric behavior
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Tunable constants — keep in ONE place; Kotlin port mirrors these names.
WORK_WIDTH = 1000             #  target working width; images are resized to this smaller images remain unchanged
ADAPTIVE_BLOCK_SIZE = 31      # odd; neighborhood for adaptive threshold
ADAPTIVE_C = 15               # constant subtracted from local mean
LINE_INK_THRESHOLD = 0.005    # row counts as "ink" if >0.5% white pixels
MIN_LINE_HEIGHT_PX = 8        # discard line bands thinner than this
MIN_COMPONENT_AREA = 15       # discard specks smaller than this (px)
MIN_COMPONENT_DIM = 3         # discard components thinner than this (px)
CHAR_MARGIN_FRAC = 0.15       # margin added around glyph before resize
EMNIST_SIZE = 28
SPACE_GAP_FACTOR = 1.7        # gap > factor * median gap  -> word space


@dataclass
class CharBox:
    """A segmented character: bounding box in line coordinates."""
    x: int
    y: int
    w: int
    h: int

# Stage 0: resolution normalization
def normalize_size(img: np.ndarray) -> np.ndarray:
    """
    Scale the image down to a fixed working width.

    Every threshold below (ADAPTIVE_BLOCK_SIZE, MIN_COMPONENT_AREA,
    MIN_LINE_HEIGHT_PX, MIN_COMPONENT_DIM) is expressed in absolute pixels.
    Without this stage their meaning changes with camera resolution: at
    ~3500 px the adaptive-threshold window is narrower than a pen stroke,
    so thick strokes hollow out and fragment into vertical slivers.
    Never upscales: smaller inputs (e.g. the synthetic smoke test) pass through.
    """
    h, w = img.shape[:2]
    if w <= WORK_WIDTH:
        return img
    scale = WORK_WIDTH / w
    return cv2.resize(img, (WORK_WIDTH, int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)

# Stage 1-2: grayscale + binarization
def to_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def binarize(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive (local) threshold -> robust to uneven lighting, the main
    real-photo failure mode of a global threshold.
    THRESH_BINARY_INV: dark ink on light paper becomes WHITE on BLACK,
    matching EMNIST polarity. Output: uint8 {0, 255}.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)


# Stage 3: line segmentation (horizontal projection profile)
def segment_lines(binary: np.ndarray) -> list[tuple[int, int]]:
    """
    Returns [(y0, y1), ...] bands containing text lines, top to bottom.
    A row is 'inked' if its white-pixel fraction exceeds LINE_INK_THRESHOLD;
    consecutive inked rows form a band; thin bands are discarded as noise.
    """
    row_ink = (binary > 0).mean(axis=1)
    inked = row_ink > LINE_INK_THRESHOLD

    bands = []
    start = None
    for y, on in enumerate(inked):
        if on and start is None:
            start = y
        elif not on and start is not None:
            if y - start >= MIN_LINE_HEIGHT_PX:
                bands.append((start, y))
            start = None
    if start is not None and len(inked) - start >= MIN_LINE_HEIGHT_PX:
        bands.append((start, len(inked)))
    return bands


# Stage 4: character segmentation within a line band
def _merge_vertical(boxes: list[CharBox]) -> list[CharBox]:
    """
    Merge components that overlap horizontally (>=50% of the narrower box):
    the dot of 'i'/'j' is a separate component above the stem and must be
    re-united with it before classification.
    """
    boxes = sorted(boxes, key=lambda b: b.x)
    merged: list[CharBox] = []
    for b in boxes:
        if merged:
            m = merged[-1]
            overlap = min(m.x + m.w, b.x + b.w) - max(m.x, b.x)
            if overlap > 0 and overlap >= 0.5 * min(m.w, b.w):
                x0 = min(m.x, b.x); y0 = min(m.y, b.y)
                x1 = max(m.x + m.w, b.x + b.w); y1 = max(m.y + m.h, b.y + b.h)
                merged[-1] = CharBox(x0, y0, x1 - x0, y1 - y0)
                continue
        merged.append(b)
    return merged


def segment_characters(line_binary: np.ndarray) -> list[CharBox]:
    """
    Connected components -> filter specks -> merge i/j dots -> sort by x.
    Input: binary crop of ONE line band. Output: boxes in line coordinates.
    """
    n, _, stats, _ = cv2.connectedComponentsWithStats(line_binary, connectivity=8)
    boxes = []
    for i in range(1, n):  # 0 is background
        x, y, w, h, area = stats[i]
        if area < MIN_COMPONENT_AREA or w < MIN_COMPONENT_DIM or h < MIN_COMPONENT_DIM:
            continue
        boxes.append(CharBox(int(x), int(y), int(w), int(h)))
    return _merge_vertical(boxes)


# Stage 5: crop -> EMNIST-style 28x28 float tensor
def to_emnist(glyph_binary: np.ndarray) -> np.ndarray:
    """
    Input : binary crop (uint8 {0,255}) tightly around one glyph.
    Output: float32 (28, 28, 1) in [0,1], white-on-black, glyph centered
            in a square canvas with proportional margin, mildly blurred to
            match EMNIST stroke softness.
    Every step must stay portable: pad -> resize -> blur -> scale.
    """
    h, w = glyph_binary.shape
    side = max(h, w)
    margin = int(round(side * CHAR_MARGIN_FRAC))
    canvas_side = side + 2 * margin

    top = (canvas_side - h) // 2
    bottom = canvas_side - h - top
    left = (canvas_side - w) // 2
    right = canvas_side - w - left
    canvas = cv2.copyMakeBorder(glyph_binary, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=0)

    resized = cv2.resize(canvas, (EMNIST_SIZE, EMNIST_SIZE),
                         interpolation=cv2.INTER_AREA)
    out = resized.astype(np.float32) / 255.0
    return out[..., None]


# Word spacing: decide where spaces go from inter-character gaps
def detect_spaces(boxes: list[CharBox]) -> list[bool]:
    """
    Returns flags[i] == True if a SPACE precedes character i (i>=1).
    Rule: gap between consecutive boxes > SPACE_GAP_FACTOR * median gap.
    Robust to font size because it is relative to the line's own gaps.
    """
    if len(boxes) < 2:
        return [False] * len(boxes)
    gaps = [boxes[i].x - (boxes[i-1].x + boxes[i-1].w) for i in range(1, len(boxes))]
    positive = sorted(g for g in gaps if g > 0)
    if not positive:
        return [False] * len(boxes)
    median_gap = positive[len(positive) // 2]
    threshold = max(SPACE_GAP_FACTOR * median_gap, 1.0)
    return [False] + [g > threshold for g in gaps]
