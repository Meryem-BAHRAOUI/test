from __future__ import annotations

from typing import Any

from geometry import (
    area,
    bbox_distance,
    below_search_zone,
    center_as_point,
    contains,
    height,
    horizontal_gap_from_right,
    intersects,
    intersection_area,
    is_field_like,
    is_square_like,
    left_gap,
    overlap_width,
    right_search_zone,
    vertical_center_delta,
    vertical_gap_from_below,
)
from schemas import ActionRequest, OCRBlock, ScreenData, UIElement  # noqa: F401 (UIElement used in fallback)
from text_matcher import find_best_ocr_match


def resolve_actions(screen: ScreenData, actions: list[ActionRequest]) -> dict[str, Any]:
    resolved_actions: list[dict[str, Any]] = []
    unresolved_actions: list[dict[str, Any]] = []

    for action in actions:
        text_block = find_best_ocr_match(action.target_text, screen.ocr_blocks)
        if text_block is None:
            unresolved_actions.append(
                _build_unresolved_result(
                    action,
                    reason="no_text_match_found",
                )
            )
            continue

        resolved = resolve_action(screen, action, text_block)
        if resolved is None:
            unresolved_actions.append(
                _build_unresolved_result(
                    action,
                    reason="no_ui_element_matched",
                    matched_text_bbox=text_block.bbox,
                )
            )
            continue

        resolved_actions.append(resolved)

    return {
        "resolved_actions": resolved_actions,
        "unresolved_actions": unresolved_actions,
    }


def resolve_action(
    screen: ScreenData,
    action: ActionRequest,
    text_block: OCRBlock,
) -> dict[str, Any] | None:
    operation = action.action.strip().lower()

    if operation in {"fill", "select"}:
        element = _resolve_field_like_target(text_block, screen)
        if element is None:
            return None
        return _build_resolved_result(
            action=action,
            matched_text_bbox=text_block.bbox,
            resolved_element=element,
            resolved_bbox=element.bbox,
        )

    if operation in {"check", "uncheck"}:
        element = _resolve_checkbox_or_radio(text_block, screen.ui_elements)
        if element is None:
            return None
        return _build_resolved_result(
            action=action,
            matched_text_bbox=text_block.bbox,
            resolved_element=element,
            resolved_bbox=element.bbox,
        )

    if operation == "click":
        element = _resolve_click_button_target(text_block, screen.ui_elements)
        if element is not None:
            return _build_resolved_result(
                action=action,
                matched_text_bbox=text_block.bbox,
                resolved_element=element,
                resolved_bbox=element.bbox,
            )
        return _build_resolved_result(
            action=action,
            matched_text_bbox=text_block.bbox,
            resolved_element=None,
            resolved_bbox=text_block.bbox,
        )

    return None


def _resolve_click_button_target(
    text_block: OCRBlock,
    ui_elements: list[UIElement],
) -> UIElement | None:
    containing = [element for element in ui_elements if contains(element.bbox, text_block.bbox)]
    if containing:
        return sorted(
            containing,
            key=lambda element: (area(element.bbox), element.bbox[1], element.bbox[0]),
        )[0]

    overlapping = [
        element for element in ui_elements if intersection_area(element.bbox, text_block.bbox) > 0
    ]
    if overlapping:
        return sorted(
            overlapping,
            key=lambda element: (
                -intersection_area(element.bbox, text_block.bbox),
                bbox_distance(element.bbox, text_block.bbox),
                element.bbox[1],
                element.bbox[0],
            ),
        )[0]

    return None


def _resolve_checkbox_or_radio(
    text_block: OCRBlock,
    ui_elements: list[UIElement],
) -> UIElement | None:
    text_bbox = text_block.bbox
    alignment_limit = max(20, text_bbox[3] - text_bbox[1])

    candidates = [
        element
        for element in ui_elements
        if element.bbox[2] <= text_bbox[0] + 8
        and vertical_center_delta(element.bbox, text_bbox) <= alignment_limit
    ]
    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda element: (
            left_gap(text_bbox, element.bbox),
            vertical_center_delta(text_bbox, element.bbox),
            0 if is_square_like(element.bbox) else 1,
            area(element.bbox),
            element.bbox[0],
        ),
    )[0]


def _resolve_field_like_target(
    text_block: OCRBlock,
    screen: ScreenData,
) -> UIElement | None:
    text_h  = height(text_block.bbox)
    text_cy = (text_block.bbox[1] + text_block.bbox[3]) / 2.0

    right_candidates = _filter_right_candidates(text_block, screen)
    if right_candidates:

        def _right_sort(element: UIElement):
            el_cy      = (element.bbox[1] + element.bbox[3]) / 2.0
            el_h       = height(element.bbox)
            vert_delta = abs(el_cy - text_cy)

            # Bucket 0 = même ligne, 1 = ligne adjacente, 2 = loin
            if vert_delta <= max(text_h * 0.9, 12):
                row_bucket = 0
            elif vert_delta <= max(text_h * 2.0, 30):
                row_bucket = 1
            else:
                row_bucket = 2

            # Pénalité si l'élément est bien plus haut que le texte
            # (grand conteneur qui englobe plusieurs lignes)
            ideal_h      = max(text_h * 1.8, 24)
            height_ratio = el_h / ideal_h if ideal_h > 0 else 1.0
            height_pen   = 0 if height_ratio <= 3.0 else 1  # pénalise si > 3× attendu

            return (
                row_bucket + height_pen,
                horizontal_gap_from_right(text_block.bbox, element.bbox),
                vert_delta,
                0 if is_field_like(element.bbox) else 1,
                el_h,
                -area(element.bbox),
                element.bbox[0],
                element.bbox[1],
            )

        best = sorted(right_candidates, key=_right_sort)[0]

        # Si le meilleur candidat est trop loin verticalement ET trop grand,
        # on préfère le fallback synthétique (position estimée à droite du label)
        best_cy    = (best.bbox[1] + best.bbox[3]) / 2.0
        best_h     = height(best.bbox)
        bad_match  = (
            abs(best_cy - text_cy) > max(text_h * 3, 40)
            and best_h > max(text_h * 3, 50)
        )
        if not bad_match:
            return best

    below_candidates = _filter_below_candidates(text_block, screen)
    if below_candidates:
        return sorted(
            below_candidates,
            key=lambda element: (
                vertical_gap_from_below(text_block.bbox, element.bbox),
                abs(element.bbox[0] - text_block.bbox[0]),
                0 if is_field_like(element.bbox) else 1,
                height(element.bbox),
                -area(element.bbox),
                -overlap_width(text_block.bbox, element.bbox),
                element.bbox[1],
                element.bbox[0],
            ),
        )[0]

    # Fallback synthétique : position estimée juste à droite du label
    tx1, ty1, tx2, ty2 = text_block.bbox
    sw, sh = screen.screen_size
    syn_bbox = [
        min(tx2 + 8,  sw),
        max(ty1 - 4,  0),
        min(tx2 + 260, sw),
        min(ty2 + 4,  sh),
    ]
    return UIElement(element_id=None, bbox=syn_bbox, score=None)


def _filter_right_candidates(text_block: OCRBlock, screen: ScreenData) -> list[UIElement]:
    zone = right_search_zone(text_block.bbox, screen.screen_size)
    candidates = []
    for element in screen.ui_elements:
        # L'élément doit s'étendre à droite du texte
        if element.bbox[2] <= text_block.bbox[2]:
            continue
        # Le CENTRE de l'élément doit être dans la zone (pas juste une intersection)
        el_cx = (element.bbox[0] + element.bbox[2]) / 2.0
        el_cy = (element.bbox[1] + element.bbox[3]) / 2.0
        if not (zone[0] <= el_cx <= zone[2] and zone[1] <= el_cy <= zone[3]):
            continue
        candidates.append(element)
    return candidates


def _filter_below_candidates(text_block: OCRBlock, screen: ScreenData) -> list[UIElement]:
    zone = below_search_zone(text_block.bbox, screen.screen_size)
    candidates = []
    for element in screen.ui_elements:
        if not intersects(element.bbox, zone):
            continue
        if element.bbox[3] <= text_block.bbox[3]:
            continue
        candidates.append(element)
    return candidates


def _build_resolved_result(
    action: ActionRequest,
    matched_text_bbox: list[int],
    resolved_element: UIElement | None,
    resolved_bbox: list[int],
) -> dict[str, Any]:
    result = action.to_base_result()
    result["matched_text_bbox"] = matched_text_bbox
    result["resolved_element_id"] = (
        resolved_element.element_id if resolved_element is not None else None
    )
    result["resolved_bbox"] = resolved_bbox
    result["resolved_point"] = center_as_point(resolved_bbox)
    result["status"] = "resolved"
    return result


def _build_unresolved_result(
    action: ActionRequest,
    reason: str,
    matched_text_bbox: list[int] | None = None,
) -> dict[str, Any]:
    result = action.to_base_result()
    if matched_text_bbox is not None:
        result["matched_text_bbox"] = matched_text_bbox
    result["status"] = "unresolved"
    result["reason"] = reason
    return result
