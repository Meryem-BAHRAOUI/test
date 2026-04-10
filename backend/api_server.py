#!/usr/bin/env python3
"""
Unified backend — YOLOX + PaddleOCR + Ollama action resolution
Port: 8002

Endpoints:
  GET  /health
  POST /analyze/capture   — 5s delay → screenshot → YOLOX + OCR → JSON
  POST /resolve           — screen JSON + instruction → resolved actions via Ollama
"""

from __future__ import annotations
import sys, time, base64, datetime, json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2
from PIL import ImageGrab

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

# ── YOLOX ──────────────────────────────────────────────────────────────────────
from yolox_weights.yolox_ui_s import Exp as YOLOXExp
from yolox.utils import postprocess

# ── PaddleOCR ─────────────────────────────────────────────────────────────────
from paddleocr import PaddleOCR

# ── Resolve modules ────────────────────────────────────────────────────────────
from schemas import ActionRequest, ScreenData
from resolver import resolve_actions
from instruction_to_actions import (
    DEFAULT_MODEL,
    OLLAMA_DEFAULT_URL,
    generate_actions_with_metadata,
)

# ── Constants ─────────────────────────────────────────────────────────────────
WEIGHTS       = ROOT / "yolox_weights" / "best_ckpt.pth"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CONF          = 0.45
NMS           = 0.45
MIN_SCORE     = 0.45
MIN_W         = 30
MIN_H         = 22
MIN_AREA      = 900
MAX_WH_RATIO  = 12
MIN_WH_RATIO  = 0.15
IOU_DEDUP     = 0.40
OCR_MIN_CONF  = 0.45
EXCLUDED_CLASSES  = {3}
LABEL_COVER_RATIO = 0.60

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


# ── Image helpers ─────────────────────────────────────────────────────────────

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


def _iou(a: list, b: list) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _intersection_over_a(a: list, b: list) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / max(1, (a[2]-a[0]) * (a[3]-a[1]))


def _dedup(elements: list) -> list:
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
    raw = postprocess(raw, num_classes=exp.num_classes, conf_thre=CONF, nms_thre=NMS)[0]
    if raw is None:
        return []
    candidates = []
    for d in raw.tolist():
        x1 = round(d[0] / ratio); y1 = round(d[1] / ratio)
        x2 = round(d[2] / ratio); y2 = round(d[3] / ratio)
        score  = round(float(d[4]) * float(d[5]), 4)
        cls_id = int(d[6])
        bw, bh = x2 - x1, y2 - y1
        if cls_id in EXCLUDED_CLASSES:  continue
        if score < MIN_SCORE:           continue

        # Éléments carrés (checkbox / radio) : seuils réduits
        is_square = bw > 0 and bh > 0 and max(bw, bh) / min(bw, bh) <= 2.0
        if is_square and bw >= 8 and bh >= 8:
            pass  # checkbox / radio accepté même s'il est petit
        else:
            if bw < MIN_W or bh < MIN_H:                   continue
            if bw * bh < MIN_AREA:                          continue
            if bh > 0 and (bw / bh) > MAX_WH_RATIO:        continue
            if bw > 0 and (bh / bw) > (1 / MIN_WH_RATIO):  continue

        candidates.append({"bbox": [x1, y1, x2, y2], "score": score})
    filtered = _dedup(candidates)
    return [{"id": f"u{i+1}", **el} for i, el in enumerate(filtered)]


def run_ocr(img: np.ndarray) -> list:
    engine = get_ocr()
    result = engine.ocr(img, cls=True)
    if not result or result[0] is None:
        return []
    blocks = []
    for poly, (text, conf) in result[0]:
        if float(conf) < OCR_MIN_CONF:  continue
        text = text.strip()
        if not text:                    continue
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        blocks.append({
            "text": text,
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "conf": round(float(conf), 4),
        })
    return [{"id": f"t{i+1}", **b} for i, b in enumerate(blocks)]


def _is_small_square(bbox: list) -> bool:
    """Retourne True si l'élément ressemble à une checkbox ou un radio button."""
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    if bw <= 0 or bh <= 0:
        return False
    return max(bw, bh) <= 35 and max(bw, bh) / min(bw, bh) <= 2.0


def _remove_label_detections(ui_elements: list, ocr_blocks: list) -> list:
    kept = []
    for el in ui_elements:
        # Ne jamais supprimer les checkboxes / radios (petits carrés)
        if _is_small_square(el["bbox"]):
            kept.append(el)
            continue

        el_x1, el_y1, el_x2, el_y2 = el["bbox"]
        el_w = max(1, el_x2 - el_x1)
        el_h = max(1, el_y2 - el_y1)
        el_area = el_w * el_h

        # Protéger les éléments vraiment larges : ce sont des vrais champs / dropdowns.
        # Un label WinForms bordé est typiquement < 180 px de large.
        # Un vrai input ou dropdown fait au moins 3× sa hauteur ET ≥ 180 px.
        if el_w >= max(180, 3.5 * el_h):
            kept.append(el)
            continue

        is_label = False
        total_inter = 0

        for bl in ocr_blocks:
            bl_x1, bl_y1, bl_x2, bl_y2 = bl["bbox"]
            bl_area = max(1, (bl_x2 - bl_x1) * (bl_y2 - bl_y1))

            ix1 = max(el_x1, bl_x1); iy1 = max(el_y1, bl_y1)
            ix2 = min(el_x2, bl_x2); iy2 = min(el_y2, bl_y2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue

            total_inter += inter

            # Critère 1 (original) : un bloc OCR couvre ≥ 60 % de la surface de l'élément
            if inter / el_area >= LABEL_COVER_RATIO:
                is_label = True
                break

            # Critère 2 : un bloc OCR se terminant par ":" est contenu
            # à ≥ 70 % dans l'élément → label de formulaire quasi-certain
            if bl["text"].strip().endswith(":") and inter / bl_area >= 0.70:
                is_label = True
                break

        if not is_label:
            # Critère 3 (cumulatif) : la somme des intersections couvre ≥ 60 % de l'élément
            if min(total_inter, el_area) / el_area >= LABEL_COVER_RATIO:
                is_label = True

        if not is_label:
            kept.append(el)

    return [{"id": f"u{i+1}", "bbox": el["bbox"], "score": el["score"]}
            for i, el in enumerate(kept)]


def annotate(img: np.ndarray, ui_elements: list, ocr_blocks: list) -> np.ndarray:
    out = img.copy()
    for el in ui_elements:
        x1, y1, x2, y2 = el["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(out, f"{el['score']:.2f}", (x1, max(y1-4, 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
    for bl in ocr_blocks:
        x1, y1, x2, y2 = bl["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 80), 1)
        cv2.putText(out, bl["text"][:20], (x1, max(y1-4, 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 80), 1)
    return out


def img_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def _read_img_as_b64(path: str) -> str | None:
    """Reads a saved PNG from disk and returns it as base64."""
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        return img_to_b64(img)
    except Exception:
        return None


def _crop_bbox(img: np.ndarray, bbox: list, pad: int = 2) -> np.ndarray | None:
    """Crops a bbox from img with optional padding. Returns None if empty."""
    h, w = img.shape[:2]
    x1 = max(0, bbox[0] - pad); y1 = max(0, bbox[1] - pad)
    x2 = min(w, bbox[2] + pad); y2 = min(h, bbox[3] + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def crop_elements(
    img: np.ndarray,
    ui_elements: list,
    ocr_blocks: list,
    crops_dir: Path,
    pad: int = 2,
) -> dict[str, str]:
    """
    Saves a cropped PNG per UI element and OCR block.
    Returns crops_index: {element_id → absolute file path}.
    """
    crops_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}

    for el in ui_elements:
        crop = _crop_bbox(img, el["bbox"], pad)
        if crop is None:
            continue
        score_str = f"{el['score']:.2f}" if el.get("score") is not None else "0.00"
        fname = f"ui_{el['id']}_{score_str}.png"
        fpath = crops_dir / fname
        cv2.imwrite(str(fpath), crop)
        index[el["id"]] = str(fpath)

    for bl in ocr_blocks:
        crop = _crop_bbox(img, bl["bbox"], pad)
        if crop is None:
            continue
        conf_str = f"{bl['conf']:.2f}" if bl.get("conf") is not None else "0.00"
        safe = "".join(
            c for c in bl["text"][:15] if c.isalnum() or c in " _-"
        ).strip().replace(" ", "_")
        fname = f"ocr_{bl['id']}_{conf_str}_{safe}.png"
        fpath = crops_dir / fname
        cv2.imwrite(str(fpath), crop)
        index[bl["id"]] = str(fpath)

    return index


def analyze_and_save(img: np.ndarray) -> dict:
    h, w    = img.shape[:2]
    ts      = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    ts_file = ts.replace(":", "-")
    ocr_blocks  = run_ocr(img)
    raw_ui      = detect_ui(img)
    ui_elements = _remove_label_detections(raw_ui, ocr_blocks)
    annotated   = annotate(img, ui_elements, ocr_blocks)

    # Save original image to disk (needed by /resolve for synthetic-element crops)
    orig_path = DATA_DIR / f"original_{ts_file}.png"
    cv2.imwrite(str(orig_path), img)

    # Crop and save every detected element
    crops_dir  = DATA_DIR / f"crops_{ts_file}"
    crops_index = crop_elements(img, ui_elements, ocr_blocks, crops_dir)
    print(f"  Crops saved → {crops_dir}  ({len(crops_index)} items)")

    clean = {
        "screen_size":       [w, h],
        "timestamp":         ts,
        "ui_elements":       ui_elements,
        "ocr_blocks":        ocr_blocks,
        "original_img_path": str(orig_path),
        "crops_dir":         str(crops_dir),
        "crops_index":       crops_index,
    }
    json_path = DATA_DIR / f"analysis_{ts_file}.json"
    json_path.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        **clean,
        "json_file":    str(json_path),
        "image_b64":    img_to_b64(annotated),
        "original_b64": img_to_b64(img),
    }


# ── Pydantic models ───────────────────────────────────────────────────────────

class ResolveRequest(BaseModel):
    screen_data: dict[str, Any]
    instruction: str
    model: str = DEFAULT_MODEL
    ollama_url: str = OLLAMA_DEFAULT_URL


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Screen Analyzer & Action Resolver", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "device":        DEVICE,
        "data_dir":      str(DATA_DIR),
        "ollama_url":    OLLAMA_DEFAULT_URL,
        "default_model": DEFAULT_MODEL,
    }


@app.post("/analyze/capture")
def analyze_capture():
    """5s delay → screenshot → YOLOX + OCR → returns annotated result."""
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
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))


def _attach_crops(
    resolved_actions: list[dict],
    crops_index: dict[str, str],
    original_img_path: str | None,
) -> None:
    """
    Mutates each resolved action in-place to add:
      - crop_b64  : crop of the resolved UI element (or resolved_bbox if synthetic)
      - ocr_crop_b64 : crop of the matched OCR text block
    """
    orig_img: np.ndarray | None = None
    if original_img_path:
        orig_img = cv2.imread(original_img_path)

    for action in resolved_actions:
        # ── UI element crop ──────────────────────────────────────────────────
        el_id = action.get("resolved_element_id")
        if el_id and el_id in crops_index:
            action["crop_b64"] = _read_img_as_b64(crops_index[el_id])
        elif orig_img is not None:
            # Synthetic element: crop resolved_bbox from original image
            rbbox = action.get("resolved_bbox")
            if rbbox:
                crop = _crop_bbox(orig_img, rbbox)
                action["crop_b64"] = img_to_b64(crop) if crop is not None else None
            else:
                action["crop_b64"] = None
        else:
            action["crop_b64"] = None

        # ── OCR text block crop ──────────────────────────────────────────────
        ocr_crop = None
        mtbbox = action.get("matched_text_bbox")
        if mtbbox and orig_img is not None:
            crop = _crop_bbox(orig_img, mtbbox)
            ocr_crop = img_to_b64(crop) if crop is not None else None
        action["ocr_crop_b64"] = ocr_crop


@app.post("/resolve")
def resolve(body: ResolveRequest):
    """Screen JSON + natural-language instruction → resolved actions via Ollama."""
    if not body.instruction.strip():
        raise HTTPException(400, "instruction must not be empty")
    try:
        screen = ScreenData.from_dict(body.screen_data)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Invalid screen_data: {exc}")
    try:
        actions_payload, llm_metadata = generate_actions_with_metadata(
            instruction=body.instruction,
            model=body.model,
            ollama_url=body.ollama_url,
        )
    except RuntimeError as exc:
        raise HTTPException(502, f"Ollama error: {exc}")
    raw_actions = actions_payload.get("actions", [])
    parsed: list[ActionRequest] = []
    for item in raw_actions:
        try:
            parsed.append(ActionRequest.from_dict(item))
        except (KeyError, ValueError) as exc:
            raise HTTPException(422, f"Invalid action from LLM: {exc}")
    result = resolve_actions(screen, parsed)

    # Attach crop images to each resolved action
    _attach_crops(
        resolved_actions=result["resolved_actions"],
        crops_index=body.screen_data.get("crops_index", {}),
        original_img_path=body.screen_data.get("original_img_path"),
    )

    return {
        "resolved_actions":   result["resolved_actions"],
        "unresolved_actions": result["unresolved_actions"],
        "generated_actions":  raw_actions,
        "llm_metadata":       llm_metadata,
    }


if __name__ == "__main__":
    print(f"Starting Unified Backend on http://127.0.0.1:8002  (device={DEVICE})")
    print(f"  Ollama : {OLLAMA_DEFAULT_URL}  model={DEFAULT_MODEL}")
    print(f"  Data   : {DATA_DIR}")
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
