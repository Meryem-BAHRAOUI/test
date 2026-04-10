# UI Automation — Screen Analyzer & Action Resolver

Automated UI testing tool that captures a screenshot, detects interface elements (YOLOX + PaddleOCR), interprets a natural-language instruction (Ollama/Qwen2.5), resolves each action to a precise screen coordinate, and crops the matched elements for robot consumption.

---

## Project Structure

```
TEST_V2/
├── backend/
│   ├── api_server.py            # FastAPI unified backend (port 8002)
│   ├── resolver.py              # Spatial action resolver
│   ├── schemas.py               # Data models (UIElement, OCRBlock, ScreenData, ActionRequest)
│   ├── geometry.py              # Spatial helpers (search zones, distances, bbox ops)
│   ├── text_matcher.py          # Fuzzy OCR text matching (difflib)
│   ├── instruction_to_actions.py# Ollama LLM client + JSON schema enforcement
│   └── yolox_weights/
│       ├── best_ckpt.pth        # Trained YOLOX-S weights (not in git)
│       └── yolox_ui_s.py        # YOLOX experiment config (6 classes, 1280×1280)
├── frontend/
│   └── src/
│       └── pages/Index.tsx      # React chat UI with SVG overlay
├── YOLOX/                       # YOLOX source (not in git)
├── start_backend.bat            # Launch backend
├── start_frontend.bat           # Launch frontend
└── ARCHITECTURE.md              # Detailed workflow & architecture
```

---

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.11 | venv recommended |
| CUDA (optional) | 11.x / 12.x | Falls back to CPU |
| Node.js | 18+ | For frontend |
| Ollama | latest | Must be running locally |
| Qwen2.5 model | 1.5b-instruct | `ollama pull qwen2.5:1.5b-instruct` |

Python packages: `fastapi`, `uvicorn`, `torch`, `opencv-python`, `paddleocr`, `Pillow`, `numpy`, `pydantic`

---

## Quick Start

### 1 — Pull the Qwen2.5 model (once)
```bash
ollama pull qwen2.5:1.5b-instruct
```

### 2 — Start the backend
```bash
start_backend.bat
# or manually:
cd backend && python api_server.py
```
Backend available at `http://127.0.0.1:8002`

### 3 — Start the frontend
```bash
start_frontend.bat
# or manually:
cd frontend && npm install && npm run dev
```
Frontend available at `http://localhost:8080`

### 4 — Use the chat interface
1. Open `http://localhost:8080`
2. Click **Capture** — you have 5 seconds to switch to the target window
3. Type your instruction, e.g.: `Fill Name with John, click Save`
4. The overlay shows each resolved action color-coded on the screenshot

---

## API Endpoints

### `GET /health`
Returns backend status, device (CPU/CUDA), Ollama URL and default model.

### `POST /analyze/capture`
5-second delay → screenshot → YOLOX detection → OCR → saves crops.

**Response includes:**
- `ui_elements` — detected UI elements with bounding boxes
- `ocr_blocks` — detected text blocks with bounding boxes
- `crops_index` — map of `{element_id → crop file path}`
- `original_img_path` — path to the saved original screenshot
- `image_b64` — annotated screenshot (base64 PNG)
- `original_b64` — clean screenshot (base64 PNG)

### `POST /resolve`
Natural-language instruction → LLM → spatial resolver → resolved actions with crops.

**Request body:**
```json
{
  "instruction": "Fill Name with John, click Save",
  "screen_data": { ... }
}
```

**Response includes per resolved action:**
- `resolved_bbox` — exact bounding box to interact with
- `resolved_point` — center point `[x, y]` for the robot to click/type
- `crop_b64` — cropped image of the resolved UI element
- `ocr_crop_b64` — cropped image of the matched label text

---

## Supported Actions

| Action | Description | Requires `value` |
|---|---|---|
| `fill` | Type text into an input field | Yes |
| `select` | Choose an option from a dropdown | Yes |
| `click` | Click a button or link | No |
| `check` | Check a checkbox or radio button | No |
| `uncheck` | Uncheck a checkbox | No |

---

## Output Files (per capture session)

```
backend/data/
├── original_<timestamp>.png        # Full screenshot
├── analysis_<timestamp>.json       # Full detection result + crops_index
└── crops_<timestamp>/
    ├── ui_u1_0.93.png              # Cropped UI elements (YOLOX)
    ├── ui_u2_0.87.png
    ├── ocr_t1_0.95_Name.png        # Cropped text blocks (OCR)
    └── ...
```
