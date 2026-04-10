from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BBox = list[int]
Point = list[int]
SUPPORTED_ACTIONS = {"fill", "click", "select", "check", "uncheck"}


@dataclass(slots=True)
class UIElement:
    element_id: str
    bbox: BBox
    score: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UIElement":
        return cls(
            element_id=str(payload["id"]),
            bbox=[int(v) for v in payload["bbox"]],
            score=float(payload["score"]) if payload.get("score") is not None else None,
        )


@dataclass(slots=True)
class OCRBlock:
    block_id: str
    text: str
    bbox: BBox
    conf: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OCRBlock":
        return cls(
            block_id=str(payload["id"]),
            text=str(payload["text"]),
            bbox=[int(v) for v in payload["bbox"]],
            conf=float(payload["conf"]) if payload.get("conf") is not None else None,
        )


@dataclass(slots=True)
class ScreenData:
    screen_size: tuple[int, int]
    timestamp: str | None
    ui_elements: list[UIElement]
    ocr_blocks: list[OCRBlock]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScreenData":
        return cls(
            screen_size=(int(payload["screen_size"][0]), int(payload["screen_size"][1])),
            timestamp=payload.get("timestamp"),
            ui_elements=[UIElement.from_dict(item) for item in payload.get("ui_elements", [])],
            ocr_blocks=[OCRBlock.from_dict(item) for item in payload.get("ocr_blocks", [])],
        )


@dataclass(slots=True)
class ActionRequest:
    action: str
    target_text: str
    value: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionRequest":
        action = str(payload["action"]).strip().lower()
        value = str(payload["value"]) if payload.get("value") is not None else None

        if action not in SUPPORTED_ACTIONS:
            raise ValueError(f"Unsupported action '{action}'.")
        if action in {"fill", "select"} and value is None:
            raise ValueError(f"Action '{action}' requires a value.")
        if action in {"click", "check", "uncheck"} and value is not None:
            value = None

        return cls(
            action=action,
            target_text=str(payload["target_text"]),
            value=value,
        )

    def to_base_result(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "action": self.action,
            "target_text": self.target_text,
        }
        if self.value is not None:
            result["value"] = self.value
        return result
