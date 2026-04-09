#!/usr/bin/env python3
"""
Screen capture  →  YOLOX UI detection  →  PaddleOCR text extraction
Outputs a clean JSON with:
  screen_size, timestamp, ui_elements (id/bbox/score), ocr_blocks (id/text/bbox/conf)

Usage:
    python capture_analyze.py              # saves result.json
    python capture_analyze.py out.json     # saves to custom path
"""

import sys
import os
import json
import datetime

import numpy as np
import torch
import cv2
from PIL import ImageGrab

# ── YOLOX: add project root so yolox_ui_s.py is importable ────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from yolox_weights.yolox_ui_s import Exp as YOLOXExp
from yolox.utils import postprocess

# ── PaddleOCR ─────────────────────────────────────────────────────────────────
from paddleocr import PaddleOCR

# ── Constants ─────────────────────────────────────────────────────────────────
WEIGHTS_PATH = os.path.join(ROOT, "yolox_weights", "best_ckpt.pth")
CONF_THRESH  = 0.30   # minimum objectness × class score
NMS_THRESH   = 0.45
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# 1. SCREENSHOT
# ─────────────────────────────────────────────────────────────────────────────

def capture_screen() -> np.ndarray:
    """Capture the primary screen. Returns a BGR numpy array."""
    pil_img = ImageGrab.grab()
    img_rgb = np.array(pil_img)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# 2. YOLOX DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(img: np.ndarray, input_size: tuple):
    """
    Letterbox-resize the image to input_size (H, W).
    Returns a float32 NCHW tensor and the scale ratio used.
    """
    ih, iw = input_size
    h, w   = img.shape[:2]
    ratio  = min(ih / h, iw / w)
    rh, rw = int(h * ratio), int(w * ratio)

    padded = np.full((ih, iw, 3), 114, dtype=np.uint8)
    padded[:rh, :rw] = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(
        np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
    ).unsqueeze(0)  # 1 × C × H × W
    return tensor, ratio


def load_yolox(weights_path: str = WEIGHTS_PATH):
    """Build the YOLOX-S model, load the checkpoint and set to eval mode."""
    exp   = YOLOXExp()
    model = exp.get_model()
    ckpt  = torch.load(weights_path, map_location=DEVICE)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, exp


def detect_ui(model, exp, img: np.ndarray) -> list:
    """
    Run YOLOX inference on img.
    Returns a list of dicts: {id, bbox [x1,y1,x2,y2], score}.
    """
    tensor, ratio = _preprocess(img, exp.test_size)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        raw = model(tensor)

    raw = postprocess(
        raw,
        num_classes=exp.num_classes,
        conf_thre=CONF_THRESH,
        nms_thre=NMS_THRESH,
    )[0]  # batch size is 1

    elements = []
    if raw is None:
        return elements

    for i, det in enumerate(raw.tolist()):
        x1, y1, x2, y2, obj_conf, cls_conf = det[:6]
        score = obj_conf * cls_conf
        elements.append({
            "id":    f"u{i + 1}",
            "bbox":  [round(x1 / ratio), round(y1 / ratio),
                      round(x2 / ratio), round(y2 / ratio)],
            "score": round(score, 4),
        })

    return elements


# ─────────────────────────────────────────────────────────────────────────────
# 3. PADDLE OCR
# ─────────────────────────────────────────────────────────────────────────────

def run_ocr(img: np.ndarray, engine: PaddleOCR) -> list:
    """
    Run PaddleOCR on img (BGR).
    Returns a list of dicts: {id, text, bbox [x1,y1,x2,y2], conf}.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result  = engine.ocr(img_rgb, cls=True)

    blocks = []
    if not result or result[0] is None:
        return blocks

    for idx, line in enumerate(result[0], start=1):
        poly, (text, conf) = line
        # poly = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        blocks.append({
            "id":   f"t{idx}",
            "text": text,
            "bbox": [round(min(xs)), round(min(ys)),
                     round(max(xs)), round(max(ys))],
            "conf": round(float(conf), 4),
        })

    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(output_path: str = "result.json") -> dict:
    # ── Step 1: screenshot ────────────────────────────────────────────────────
    print("[1/4] Capturing screen …")
    img       = capture_screen()
    h, w      = img.shape[:2]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # ── Step 2: YOLOX ─────────────────────────────────────────────────────────
    print(f"[2/4] Loading YOLOX model from {WEIGHTS_PATH} …")
    model, exp  = load_yolox()
    print("[3/4] Detecting UI elements …")
    ui_elements = detect_ui(model, exp, img)
    print(f"      → {len(ui_elements)} element(s) detected.")

    # ── Step 3: PaddleOCR ─────────────────────────────────────────────────────
    print("[4/4] Running PaddleOCR …")
    ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    ocr_blocks = run_ocr(img, ocr_engine)
    print(f"      → {len(ocr_blocks)} text block(s) extracted.")

    # ── Step 4: build and save JSON ───────────────────────────────────────────
    output = {
        "screen_size": [w, h],
        "timestamp":   timestamp,
        "ui_elements": ui_elements,
        "ocr_blocks":  ocr_blocks,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved → {os.path.abspath(output_path)}")
    return output


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "result.json"
    main(out_file)
