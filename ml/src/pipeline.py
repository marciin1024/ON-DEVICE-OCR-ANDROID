"""
Phase 4 — pipeline.py — photo -> transcription, end to end.

Glues preprocess.py stages to the int8 TFLite model. Mirrors exactly what
the Android app will do per frame. Returns rich results (boxes, crops,
confidences) so the notebook can visualize every stage and build the
failure gallery.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import preprocess as pp

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
LETTERS = "abcdefghijklmnopqrstuvwxyz"


@dataclass
class CharResult:
    box: pp.CharBox          # in full-image coordinates
    tensor: np.ndarray       # (28,28,1) float32 — what the model saw
    letter: str
    confidence: float
    space_before: bool


@dataclass
class LineResult:
    band: tuple[int, int]
    chars: list[CharResult] = field(default_factory=list)

    @property
    def text(self) -> str:
        out = []
        for c in self.chars:
            if c.space_before:
                out.append(" ")
            out.append(c.letter)
        return "".join(out)


@dataclass
class PageResult:
    binary: np.ndarray
    lines: list[LineResult] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)


class Recognizer:
    """Loads model_int8.tflite once; classify() runs one 28x28 crop."""

    def __init__(self, model_path: Path = ARTIFACTS / "model_int8.tflite"):
        self.interp = tf.lite.Interpreter(model_path=str(model_path))
        self.interp.allocate_tensors()
        self._in = self.interp.get_input_details()[0]
        self._out = self.interp.get_output_details()[0]

    def classify(self, tensor: np.ndarray) -> tuple[str, float]:
        self.interp.set_tensor(self._in["index"],
                               tensor[None].astype(self._in["dtype"]))
        self.interp.invoke()
        probs = self.interp.get_tensor(self._out["index"])[0]
        k = int(probs.argmax())
        return LETTERS[k], float(probs[k])


def transcribe(img_bgr: np.ndarray, recognizer: Recognizer) -> PageResult:
    """Full pipeline on one photo. See preprocess.py for stage semantics."""
    img_bgr = pp.normalize_size(img_bgr)
    gray = pp.to_grayscale(img_bgr)
    binary = pp.binarize(gray)
    page = PageResult(binary=binary)

    for (y0, y1) in pp.segment_lines(binary):
        line_bin = binary[y0:y1]
        boxes = pp.segment_characters(line_bin)
        if not boxes:
            continue
        spaces = pp.detect_spaces(boxes)
        line = LineResult(band=(y0, y1))
        for box, sp in zip(boxes, spaces):
            glyph = line_bin[box.y:box.y + box.h, box.x:box.x + box.w]
            tensor = pp.to_emnist(glyph)
            letter, conf = recognizer.classify(tensor)
            global_box = pp.CharBox(box.x, box.y + y0, box.w, box.h)
            line.chars.append(CharResult(global_box, tensor, letter, conf, sp))
        page.lines.append(line)
    return page
