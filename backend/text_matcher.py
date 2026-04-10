from __future__ import annotations

import difflib
import string
import unicodedata

from schemas import OCRBlock


PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    lowered = text.strip().lower()
    no_accents = "".join(
        char
        for char in unicodedata.normalize("NFKD", lowered)
        if not unicodedata.combining(char)
    )
    no_punctuation = no_accents.translate(PUNCTUATION_TABLE)
    collapsed = " ".join(no_punctuation.split())
    return collapsed


def similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left, right).ratio()


def find_best_ocr_match(
    target_text: str,
    ocr_blocks: list[OCRBlock],
    fuzzy_threshold: float = 0.72,
) -> OCRBlock | None:
    normalized_target = normalize_text(target_text)

    exact_matches = [
        block for block in ocr_blocks if normalize_text(block.text) == normalized_target
    ]
    if exact_matches:
        return sorted(
            exact_matches,
            key=lambda block: (
                -(block.conf if block.conf is not None else 0.0),
                block.bbox[1],
                block.bbox[0],
            ),
        )[0]

    scored_matches: list[tuple[float, OCRBlock]] = []
    for block in ocr_blocks:
        normalized_block = normalize_text(block.text)
        ratio = similarity(normalized_target, normalized_block)
        if normalized_target and normalized_target in normalized_block:
            ratio = max(ratio, 0.88)
        elif normalized_block and normalized_block in normalized_target:
            ratio = max(ratio, 0.84)

        if ratio >= fuzzy_threshold:
            scored_matches.append((ratio, block))

    if not scored_matches:
        return None

    scored_matches.sort(
        key=lambda item: (
            -item[0],
            -(item[1].conf if item[1].conf is not None else 0.0),
            item[1].bbox[1],
            item[1].bbox[0],
        )
    )
    return scored_matches[0][1]
