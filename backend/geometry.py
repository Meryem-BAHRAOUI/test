from __future__ import annotations

import math

from schemas import BBox


def width(bbox: BBox) -> int:
    return bbox[2] - bbox[0]


def height(bbox: BBox) -> int:
    return bbox[3] - bbox[1]


def center(bbox: BBox) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def center_as_point(bbox: BBox) -> list[int]:
    cx, cy = center(bbox)
    return [round(cx), round(cy)]


def area(bbox: BBox) -> int:
    return max(0, width(bbox)) * max(0, height(bbox))


def contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def intersection_area(a: BBox, b: BBox) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def intersects(a: BBox, b: BBox) -> bool:
    return intersection_area(a, b) > 0


def bbox_distance(a: BBox, b: BBox) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return math.hypot(dx, dy)


def clamp_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    screen_width: int,
    screen_height: int,
) -> BBox:
    return [
        max(0, round(x1)),
        max(0, round(y1)),
        min(screen_width, round(x2)),
        min(screen_height, round(y2)),
    ]


def right_search_zone(text_bbox: BBox, screen_size: tuple[int, int]) -> BBox:
    screen_width, screen_height = screen_size
    text_height = height(text_bbox)
    return clamp_bbox(
        x1=text_bbox[2] + 4,
        y1=text_bbox[1] - max(20, 1.5 * text_height),
        x2=text_bbox[2] + min(450, 0.35 * screen_width),
        y2=text_bbox[3] + max(20, 1.5 * text_height),
        screen_width=screen_width,
        screen_height=screen_height,
    )


def below_search_zone(text_bbox: BBox, screen_size: tuple[int, int]) -> BBox:
    screen_width, screen_height = screen_size
    text_width = width(text_bbox)
    return clamp_bbox(
        x1=text_bbox[0] - 0.5 * text_width,
        y1=text_bbox[3] + 4,
        x2=text_bbox[2] + 3.0 * text_width,
        y2=text_bbox[3] + min(250, 0.30 * screen_height),
        screen_width=screen_width,
        screen_height=screen_height,
    )


def vertical_center_delta(a: BBox, b: BBox) -> float:
    return abs(center(a)[1] - center(b)[1])


def horizontal_center_delta(a: BBox, b: BBox) -> float:
    return abs(center(a)[0] - center(b)[0])


def horizontal_gap_from_right(text_bbox: BBox, element_bbox: BBox) -> int:
    return max(0, element_bbox[0] - text_bbox[2])


def vertical_gap_from_below(text_bbox: BBox, element_bbox: BBox) -> int:
    return max(0, element_bbox[1] - text_bbox[3])


def left_gap(text_bbox: BBox, element_bbox: BBox) -> int:
    return max(0, text_bbox[0] - element_bbox[2])


def overlap_width(a: BBox, b: BBox) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0]))


def is_square_like(bbox: BBox) -> bool:
    w = max(1, width(bbox))
    h = max(1, height(bbox))
    ratio = max(w, h) / min(w, h)
    return ratio <= 1.4


def is_field_like(bbox: BBox) -> bool:
    w = max(1, width(bbox))
    h = max(1, height(bbox))
    return w >= max(60, 1.6 * h)
