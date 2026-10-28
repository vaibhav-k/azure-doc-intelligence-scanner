"""
OCR-engine confidence assessment for Azure Document Intelligence results.

Scope: this module measures *OCR* confidence only (Document Intelligence word scores). It is
deliberately separate from LLM extraction, whose output carries no calibrated probability.

Rules
-----
* Missing, non-numeric, NaN or out-of-range confidence is UNKNOWN and is flagged; it is never
  treated as 1.0 and no substitute number is invented.
* A page that exposes lines but no scored words is ``unscored_page`` (Document Intelligence v4
  lines carry no confidence), so it cannot silently pass.
* Empty or structurally incomplete results are flagged (``no_pages`` / ``no_text``).
* ``ReviewItem.content`` holds OCR text (potential PHI) and is excluded from ``repr``.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any

DEFAULT_OCR_CONFIDENCE_THRESHOLD = 0.85


class OcrFlag(str, enum.Enum):
    LOW_CONFIDENCE = "low_confidence"
    MISSING_CONFIDENCE = "missing_confidence"
    INVALID_CONFIDENCE = "invalid_confidence"
    UNSCORED_PAGE = "unscored_page"
    NO_PAGES = "no_pages"
    NO_TEXT = "no_text"


@dataclass(frozen=True)
class OcrWord:
    page: int
    content: str = field(repr=False)
    confidence: float | None  # None == unknown (never defaulted)
    span_offset: int | None
    span_length: int | None
    polygon: tuple[float, ...] | None = field(default=None, repr=False)


@dataclass(frozen=True)
class ReviewItem:
    page: int
    flag: OcrFlag
    confidence: float | None
    content: str = field(default="", repr=False)
    span_offset: int | None = None
    span_length: int | None = None
    polygon: tuple[float, ...] | None = field(default=None, repr=False)


@dataclass(frozen=True)
class OcrConfidenceReport:
    threshold: float
    words: tuple[OcrWord, ...]
    review_items: tuple[ReviewItem, ...]
    page_count: int
    scored_word_count: int
    min_confidence: float | None
    mean_confidence: float | None

    @property
    def needs_review(self) -> bool:
        return bool(self.review_items)

    @property
    def flags(self) -> frozenset[OcrFlag]:
        return frozenset(i.flag for i in self.review_items)

    def flag_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.review_items:
            counts[item.flag.value] = counts.get(item.flag.value, 0) + 1
        return counts


def _item(word: OcrWord, flag: OcrFlag, confidence: float | None) -> ReviewItem:
    return ReviewItem(
        page=word.page,
        flag=flag,
        confidence=confidence,
        content=word.content,
        span_offset=word.span_offset,
        span_length=word.span_length,
        polygon=word.polygon,
    )


def validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, int | float) or not 0.0 < threshold <= 1.0:
        raise ValueError("ocr confidence threshold must be a number in (0, 1]")
    return float(threshold)


def _read_confidence(raw: Any) -> tuple[float | None, OcrFlag | None]:
    if raw is None:
        return None, OcrFlag.MISSING_CONFIDENCE
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None, OcrFlag.INVALID_CONFIDENCE
    value = float(raw)
    if math.isnan(value) or value < 0.0 or value > 1.0:
        return None, OcrFlag.INVALID_CONFIDENCE
    return value, None


def _page_number(page: Any, index: int) -> int:
    n = getattr(page, "page_number", None)
    return n if isinstance(n, int) and not isinstance(n, bool) else index + 1


def _process_word(
    word: Any, page_no: int, threshold: float
) -> tuple[OcrWord, list[ReviewItem], float | None]:
    content = getattr(word, "content", None) or ""
    span = getattr(word, "span", None)
    poly = getattr(word, "polygon", None)
    offset = getattr(span, "offset", None)
    length = getattr(span, "length", None)

    value, problem = _read_confidence(getattr(word, "confidence", None))

    ocr_word = OcrWord(
        page=page_no,
        content=content,
        confidence=value,
        span_offset=offset if isinstance(offset, int) else None,
        span_length=length if isinstance(length, int) else None,
        polygon=tuple(float(p) for p in poly) if poly else None,
    )

    item_list: list[ReviewItem] = []
    if problem is not None:
        item_list.append(_item(ocr_word, problem, None))
        return ocr_word, item_list, None

    if value is not None and value < threshold:
        item_list.append(_item(ocr_word, OcrFlag.LOW_CONFIDENCE, round(value, 4)))

    return ocr_word, item_list, value


def _process_page(
    page: Any, index: int, threshold: float
) -> tuple[list[OcrWord], list[ReviewItem], list[float]]:
    page_no = _page_number(page, index)
    words = list(getattr(page, "words", None) or [])

    if not words:
        has_lines = bool(getattr(page, "lines", None))
        flag = OcrFlag.UNSCORED_PAGE if has_lines else OcrFlag.NO_TEXT
        empty_items: list[ReviewItem] = [ReviewItem(page=page_no, flag=flag, confidence=None)]
        return [], empty_items, []

    words_out: list[OcrWord] = []
    items_out: list[ReviewItem] = []
    scores_out: list[float] = []

    for word in words:
        w_obj, w_items, score = _process_word(word, page_no, threshold)
        words_out.append(w_obj)
        items_out.extend(w_items)
        if score is not None:
            scores_out.append(score)

    return words_out, items_out, scores_out


def evaluate_ocr_confidence(
    result: Any, threshold: float = DEFAULT_OCR_CONFIDENCE_THRESHOLD
) -> OcrConfidenceReport:
    """Assess word-level OCR confidence. Fails closed on anything unknown."""
    threshold = validate_threshold(threshold)
    pages = list(getattr(result, "pages", None) or [])
    if not pages:
        item = ReviewItem(page=0, flag=OcrFlag.NO_PAGES, confidence=None)
        return OcrConfidenceReport(threshold, (), (item,), 0, 0, None, None)

    words_out: list[OcrWord] = []
    items: list[ReviewItem] = []
    scores: list[float] = []

    for index, page in enumerate(pages):
        p_words, p_items, p_scores = _process_page(page, index, threshold)
        words_out.extend(p_words)
        items.extend(p_items)
        scores.extend(p_scores)

    return OcrConfidenceReport(
        threshold=threshold,
        words=tuple(words_out),
        review_items=tuple(items),
        page_count=len(pages),
        scored_word_count=len(scores),
        min_confidence=min(scores) if scores else None,
        mean_confidence=(sum(scores) / len(scores)) if scores else None,
    )


def words_in_span(report: OcrConfidenceReport, offset: int, length: int) -> list[OcrWord]:
    """OCR words whose content span overlaps ``[offset, offset + length)`` in ``result.content``."""
    end = offset + length
    return [
        w
        for w in report.words
        if w.span_offset is not None
        and w.span_length is not None
        and w.span_offset < end
        and offset < w.span_offset + w.span_length
    ]
