# Architecture & Workflow

## Overview

```
User Instruction (natural language)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                      PIPELINE                                  │
│                                                               │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │ Screenshot│───▶│ YOLOX-S      │    │ PaddleOCR          │  │
│  │ (PIL Grab)│    │ UI Detection │    │ Text Extraction    │  │
│  └──────────┘    │ 6 classes    │    │ (angle cls, en)    │  │
│                  │ 1280×1280    │    └────────┬───────────┘  │
│                  └──────┬───────┘             │              │
│                         │ ui_elements         │ ocr_blocks   │
│                         ▼                     ▼              │
│                  ┌─────────────────────────────────────┐     │
│                  │   Label Filter (_remove_labels)      │     │
│                  │   + Crop & Save (crops_index)        │     │
│                  └──────────────┬──────────────────────┘     │
│                                 │ screen_data JSON            │
│                                 ▼                             │
│                  ┌──────────────────────────┐                │
│                  │   Ollama / Qwen2.5        │                │
│                  │   Instruction → Actions   │                │
│                  │   JSON schema enforced    │                │
│                  └──────────────┬────────────┘               │
│                                 │ [{action, target_text, value}]
│                                 ▼                             │
│                  ┌──────────────────────────┐                │
│                  │   Spatial Resolver        │                │
│                  │   text_matcher → OCR block│                │
│                  │   right/below search zone │                │
│                  │   → resolved_bbox + point │                │
│                  └──────────────┬────────────┘               │
│                                 │                             │
│                                 ▼                             │
│                  ┌──────────────────────────┐                │
│                  │   Crop Attachment         │                │
│                  │   crop_b64 + ocr_crop_b64 │                │
│                  └──────────────┬────────────┘               │
└─────────────────────────────────┼─────────────────────────────┘
                                  │
                                  ▼
                    Robot / Automation Engine
                    (resolved_point, crop_b64)
```

---

## Module Breakdown

### `api_server.py` — Unified Backend

Entry point. Orchestrates the full pipeline.

**Key functions:**

| Function | Role |
|---|---|
| `detect_ui(img)` | Runs YOLOX inference, filters by score/size, deduplicates by IoU |
| `run_ocr(img)` | Runs PaddleOCR, filters by confidence, returns text blocks |
| `_remove_label_detections(ui, ocr)` | Removes YOLOX boxes that are form labels (3 criteria below) |
| `crop_elements(img, ui, ocr, dir)` | Saves one PNG per element + returns `crops_index` |
| `analyze_and_save(img)` | Full capture pipeline: OCR → YOLOX → filter → crop → save |
| `_attach_crops(actions, index, path)` | Enriches resolved actions with `crop_b64` + `ocr_crop_b64` |

**Label filter criteria** (element is removed if any matches):
1. A single OCR block covers ≥ 60% of the element area
2. An OCR block ending with `:` is ≥ 70% inside the element
3. The cumulative area of all overlapping OCR blocks covers ≥ 60% of the element

**Label filter protection** (element is never removed if):
- It is a small square ≤ 35px → checkbox or radio button
- Its width ≥ max(180px, 3.5 × height) → it is a real input field or dropdown

---

### `resolver.py` — Spatial Action Resolver

Maps each `ActionRequest` (from LLM) to a `UIElement` and a pixel coordinate.

**Resolution strategy per action type:**

```
fill / select
    1. Find OCR block matching target_text (fuzzy)
    2. Search RIGHT of that block (right_search_zone)
       → center-based filter (not intersection)
       → sort by: row_bucket (same/adjacent/far) + height_penalty + gap
    3. If no right candidate → search BELOW (below_search_zone)
    4. If still nothing → synthetic bbox estimated to the right of label

check / uncheck
    1. Find OCR block matching target_text
    2. Search for square-like elements to the LEFT of text
       within vertical alignment tolerance

click
    1. Find OCR block matching target_text
    2. Find smallest containing UI element
    3. If none → use text block bbox directly
```

**Row bucket system** (prevents wrong-row field assignment):

| Bucket | Condition | Priority |
|---|---|---|
| 0 — same row | `|Δcy| ≤ max(text_h × 0.9, 12)` | Highest |
| 1 — adjacent row | `|Δcy| ≤ max(text_h × 2.0, 30)` | Medium |
| 2 — far | otherwise | Lowest |

Height penalty: elements taller than 3× the expected field height get +1 bucket penalty (avoids large containers that span multiple rows).

---

### `text_matcher.py` — Fuzzy Text Matching

Normalizes text (lowercase, strip accents, remove punctuation, collapse spaces) then:

1. **Exact match** on normalized text → sorted by confidence
2. **Substring boost**: if target ⊂ block or block ⊂ target → ratio boosted to 0.88 / 0.84
3. **Fuzzy match** via `difflib.SequenceMatcher` with threshold 0.72

---

### `geometry.py` — Search Zones

```
right_search_zone(text_bbox, screen_size):
    x1 = text.x2 + 4
    y1 = text.y1 - max(20, 1.5 × text_h)
    x2 = text.x2 + min(450, 0.35 × screen_w)
    y2 = text.y3 + max(20, 1.5 × text_h)

below_search_zone(text_bbox, screen_size):
    x1 = text.x1 - 0.5 × text_w
    y1 = text.y2 + 4
    x2 = text.x2 + 3.0 × text_w
    y2 = text.y2 + min(250, 0.30 × screen_h)
```

---

### `instruction_to_actions.py` — LLM Client

Sends the user instruction to Ollama with:
- A strict system prompt enforcing JSON-only output
- Structured output (JSON schema via Ollama `format` parameter)
- Temperature = 0, max 180 tokens
- Up to 2 attempts: if first response fails validation, a repair prompt is sent automatically

---

### `schemas.py` — Data Models

| Class | Fields |
|---|---|
| `UIElement` | `element_id`, `bbox [x1,y1,x2,y2]`, `score` |
| `OCRBlock` | `block_id`, `text`, `bbox`, `conf` |
| `ScreenData` | `screen_size`, `timestamp`, `ui_elements`, `ocr_blocks` |
| `ActionRequest` | `action`, `target_text`, `value` |

---

### `frontend/src/pages/Index.tsx` — Chat UI

React 18 + TypeScript + Vite.

**Components:**
- `ChatBubble` — renders a message (user or assistant)
- `ResolutionOverlay` — SVG layer drawn over the screenshot showing:
  - Dashed rectangle = `matched_text_bbox` (OCR label found)
  - Solid rectangle = `resolved_bbox` (element to interact with)
  - Cross = `resolved_point` (exact pixel for robot)
- Each action type has a distinct color:

| Action | Color |
|---|---|
| `fill` | Blue `#3b82f6` |
| `click` | Purple `#a855f7` |
| `select` | Orange `#f97316` |
| `check` / `uncheck` | Teal `#14b8a6` |

---

## Data Flow (sequence)

```
Browser                 FastAPI (8002)              Ollama (11434)
  │                          │                           │
  │── POST /analyze/capture ▶│                           │
  │                     sleep(5s)                        │
  │                     screenshot                       │
  │                     YOLOX inference                  │
  │                     PaddleOCR                        │
  │                     label filter                     │
  │                     crop & save                      │
  │◀── screen_data (JSON) ───│                           │
  │    + image_b64            │                           │
  │    + crops_index          │                           │
  │                          │                           │
  │── POST /resolve ─────── ▶│                           │
  │   {instruction,           │── /api/chat ────────── ▶│
  │    screen_data}           │◀── actions JSON ─────────│
  │                      text_matcher                    │
  │                      spatial resolver                │
  │                      _attach_crops                   │
  │◀── resolved_actions ─────│                           │
  │    + crop_b64             │                           │
  │    + ocr_crop_b64         │                           │
  │                          │                           │
  [SVG overlay rendered]
  [crops forwarded to robot]
```

---

## YOLOX Model

- Architecture: YOLOX-S (depth=0.33, width=0.50)
- Input size: 1280 × 1280
- Classes: 6 UI element types
- Class 3 excluded from inference results
- Inference thresholds: `conf=0.45`, `nms=0.45`, `min_score=0.45`
- Size filters: `min_w=30`, `min_h=22`, `min_area=900`
- Square bypass: elements with `max/min ≤ 2.0` and `≥ 8px` pass regardless of size filters
- IoU deduplication: threshold 0.40
