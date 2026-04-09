#!/usr/bin/env python3
"""
FastAPI server — YOLOX + PaddleOCR screen analyzer
Port: 8002

Endpoints:
  POST /analyze/capture  — 5s delay then live screenshot → YOLOX + OCR → saves JSON to data/
  GET  /health
"""

from __future__ import annotations
import sys, time, base64, datetime, json
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import ImageGrab

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── YOLOX ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))

from yolox_weights.yolox_ui_s import Exp as YOLOXExp
from yolox.utils import postprocess

# ── PaddleOCR ─────────────────────────────────────────────────────────────────
from paddleocr import PaddleOCR

# ── Constants ─────────────────────────────────────────────────────────────────
WEIGHTS = ROOT / "yolox_weights" / "best_ckpt.pth"
CONF    = 0.45   # seuil de détection YOLOX
NMS     = 0.45   # IoU pour NMS

# ── Filtres post-détection ────────────────────────────────────────────────────
MIN_SCORE    = 0.45   # score minimum conservé
MIN_W        = 30     # largeur minimale en pixels
MIN_H        = 22     # hauteur minimale en pixels
MIN_AREA     = 900    # aire minimale (w*h) en pixels²
MAX_WH_RATIO = 12     # si largeur/hauteur > 12 → barre plate → ignoré
MIN_WH_RATIO = 0.15   # si hauteur >> largeur → ignoré
IOU_DEDUP    = 0.40   # IoU au-delà duquel on supprime le doublon (score plus bas)
OCR_MIN_CONF = 0.45   # confiance OCR minimale conservée

# Classes YOLOX à exclure
EXCLUDED_CLASSES  = {3}
# Si la bbox YOLOX est couverte à plus de X% par un bloc OCR → c'est un label → ignoré
LABEL_COVER_RATIO = 0.60
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

# ── Lazy singletons ───────────────────────────────────────────────────────────
_yolox_model = None
_yolox_exp   = None
_ocr_engine  = None


def get_yolox():
    global _yolox_model, _yolox_exp
    if _yolox_model is None:
        exp   = YOLOXExp()
        model = exp.get_model()
        ckpt  = torch.load(str(WEIGHTS), map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt))
        model.to(DEVICE).eval()
        _yolox_model, _yolox_exp = model, exp
    return _yolox_model, _yolox_exp


def get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _ocr_engine


# ── Image processing ──────────────────────────────────────────────────────────

def _preprocess(img: np.ndarray, size: tuple):
    ih, iw = size
    h, w   = img.shape[:2]
    ratio  = min(ih / h, iw / w)
    rh, rw = int(h * ratio), int(w * ratio)
    padded = np.full((ih, iw, 3), 114, dtype=np.uint8)
    padded[:rh, :rw] = cv2.resize(img, (rw, rh))
    t = torch.from_numpy(
        np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
    ).unsqueeze(0)
    return t, ratio


def _intersection_over_a(a: list, b: list) -> float:
    """Ratio de l'intersection sur l'aire de A (= taux de couverture de A par B)."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (a[2]-a[0]) * (a[3]-a[1]))
    return inter / area_a


def _iou(a: list, b: list) -> float:
    """Calcule l'IoU entre deux bbox [x1,y1,x2,y2]."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def _dedup(elements: list) -> list:
    """Supprime les doublons en gardant le score le plus élevé."""
    kept = []
    for el in sorted(elements, key=lambda x: x["score"], reverse=True):
        if all(_iou(el["bbox"], k["bbox"]) < IOU_DEDUP for k in kept):
            kept.append(el)
    return kept


def detect_ui(img: np.ndarray) -> list:
    model, exp = get_yolox()
    tensor, ratio = _preprocess(img, exp.test_size)
    with torch.no_grad():
        raw = model(tensor.to(DEVICE))
    raw = postprocess(raw, num_classes=exp.num_classes,
                      conf_thre=CONF, nms_thre=NMS)[0]
    if raw is None:
        return []

    candidates = []
    for d in raw.tolist():
        x1 = round(d[0] / ratio); y1 = round(d[1] / ratio)
        x2 = round(d[2] / ratio); y2 = round(d[3] / ratio)
        score  = round(float(d[4]) * float(d[5]), 4)
        cls_id = int(d[6])
        bw = x2 - x1
        bh = y2 - y1

        # ── filtres ──────────────────────────────────────────────────────────
        if cls_id in EXCLUDED_CLASSES:                continue  # classe label exclue
        if score < MIN_SCORE:                         continue  # score trop bas
        if bw < MIN_W or bh < MIN_H:                 continue  # trop petit
        if bw * bh < MIN_AREA:                        continue  # aire trop faible
        if bh > 0 and (bw / bh) > MAX_WH_RATIO:      continue  # barre trop plate
        if bw > 0 and (bh / bw) > (1 / MIN_WH_RATIO): continue # trop étroit en largeur

        candidates.append({"bbox": [x1, y1, x2, y2], "score": score})

    # dédoublonnage par IoU
    filtered = _dedup(candidates)

    return [{"id": f"u{i+1}", **el} for i, el in enumerate(filtered)]


def run_ocr(img: np.ndarray) -> list:
    """Run PaddleOCR on BGR image. Returns list of {id, text, bbox, conf}."""
    engine = get_ocr()
    result = engine.ocr(img, cls=True)
    if not result or result[0] is None:
        return []
    blocks = []
    for poly, (text, conf) in result[0]:
        if float(conf) < OCR_MIN_CONF:    continue  # confiance trop faible
        text = text.strip()
        if not text:                       continue  # texte vide

        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        blocks.append({
            "text": text,
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "conf": round(float(conf), 4),
        })

    # renumérotation propre après filtrage
    return [{"id": f"t{i+1}", **b} for i, b in enumerate(blocks)]


def annotate(img: np.ndarray, ui_elements: list, ocr_blocks: list) -> np.ndarray:
    out = img.copy()
    for el in ui_elements:
        x1, y1, x2, y2 = el["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(out, f"{el['score']:.2f}", (x1, max(y1 - 4, 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
    for bl in ocr_blocks:
        x1, y1, x2, y2 = bl["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 80), 1)
        cv2.putText(out, bl["text"][:20], (x1, max(y1 - 4, 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 80), 1)
    return out


def img_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def _remove_label_detections(ui_elements: list, ocr_blocks: list) -> list:
    """
    Supprime les éléments YOLOX dont la surface est couverte à plus de
    LABEL_COVER_RATIO par un bloc OCR → ce sont des labels texte, pas des
    champs interactifs. PaddleOCR les détecte déjà dans ocr_blocks.
    """
    kept = []
    for el in ui_elements:
        is_label = any(
            _intersection_over_a(el["bbox"], bl["bbox"]) >= LABEL_COVER_RATIO
            for bl in ocr_blocks
        )
        if not is_label:
            kept.append(el)
    # renumérotation
    return [{"id": f"u{i+1}", "bbox": el["bbox"], "score": el["score"]}
            for i, el in enumerate(kept)]


def analyze_and_save(img: np.ndarray) -> dict:
    h, w    = img.shape[:2]
    ts      = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    ts_file = ts.replace(":", "-")

    ocr_blocks  = run_ocr(img)                              # OCR d'abord
    raw_ui      = detect_ui(img)                            # YOLOX
    ui_elements = _remove_label_detections(raw_ui, ocr_blocks)  # filtre labels
    annotated   = annotate(img, ui_elements, ocr_blocks)

    clean = {
        "screen_size": [w, h],
        "timestamp":   ts,
        "ui_elements": ui_elements,
        "ocr_blocks":  ocr_blocks,
    }
    json_path = DATA_DIR / f"analysis_{ts_file}.json"
    json_path.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        **clean,
        "json_file":    str(json_path),
        "image_b64":    img_to_b64(annotated),
        "original_b64": img_to_b64(img),
    }


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Screen Analyzer API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "data_dir": str(DATA_DIR)}


@app.post("/analyze/capture")
def analyze_capture():
    """Wait 5s, take screenshot, run YOLOX + EasyOCR, save JSON to data/."""
    try:
        print("Capture in 5 seconds — switch to your target window now…")
        time.sleep(5)

        pil = ImageGrab.grab()
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        result = analyze_and_save(img)

        print(f"  {len(result['ui_elements'])} UI elements, {len(result['ocr_blocks'])} text blocks")
        print(f"  Saved → {result['json_file']}")
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    print(f"Starting Screen Analyzer API on http://127.0.0.1:8002  (device={DEVICE})")
    print(f"JSON results will be saved to: {DATA_DIR}")
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
