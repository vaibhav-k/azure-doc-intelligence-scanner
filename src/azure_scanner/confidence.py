"""
OCR-engine confidence assessment for Azure Document Intelligence results with spatial clustering.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any

DEFAULT_OCR_CONFIDENCE_THRESHOLD = 0.85
SPATIAL_CLUSTER_DISTANCE_PX = 30.0


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
    confidence: float | None
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
    clustered_word_count: int = 1


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


def _polygon_centroid(poly: tuple[float, ...]) -> tuple[float, float] | None:
    if not poly or len(poly) < 4:
        return None
    xs = poly[0::2]
    ys = poly[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _is_clusterable(item: ReviewItem) -> bool:
    return item.flag == OcrFlag.LOW_CONFIDENCE and item.polygon is not None


def _merge_cluster_properties(page_no: int, flag: OcrFlag, cluster: list[ReviewItem]) -> ReviewItem:
    combined_content = " ".join(item.content for item in cluster if item.content)
    confidences = [item.confidence for item in cluster if item.confidence is not None]
    avg_conf = sum(confidences) / len(confidences) if confidences else None

    all_poly = [p for item in cluster if item.polygon for p in item.polygon]
    if all_poly:
        min_x, max_x = min(all_poly[0::2]), max(all_poly[0::2])
        min_y, max_y = min(all_poly[1::2]), max(all_poly[1::2])
        merged_poly = (min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y)
    else:
        merged_poly = None

    return ReviewItem(
        page=page_no,
        flag=flag,
        confidence=round(avg_conf, 4) if avg_conf is not None else None,
        content=combined_content,
        span_offset=cluster[0].span_offset,
        span_length=sum(item.span_length or 0 for item in cluster),
        polygon=merged_poly,
        clustered_word_count=len(cluster),
    )


def _is_close_to_cluster(candidate: ReviewItem, cluster: list[ReviewItem], max_dist: float) -> bool:
    c_j = _polygon_centroid(candidate.polygon or ())
    if c_j is None:
        return False
    return any(
        c_i and math.hypot(c_i[0] - c_j[0], c_i[1] - c_j[1]) <= max_dist
        for c_item in cluster
        for c_i in [_polygon_centroid(c_item.polygon or ())]
    )


def _expand_cluster(
    p_items: list[ReviewItem], used: list[bool], cluster: list[ReviewItem], max_dist: float
) -> None:
    changed = True
    while changed:
        changed = False
        for j, candidate in enumerate(p_items):
            if used[j]:
                continue
            if _is_close_to_cluster(candidate, cluster, max_dist):
                used[j] = True
                cluster.append(candidate)
                changed = True
                break


def _build_clusters_for_group(
    page_no: int, flag: OcrFlag, p_items: list[ReviewItem], max_dist: float
) -> list[ReviewItem]:
    used = [False] * len(p_items)
    clustered: list[ReviewItem] = []

    for i, base_item in enumerate(p_items):
        if used[i]:
            continue
        used[i] = True
        cluster = [base_item]

        _expand_cluster(p_items, used, cluster, max_dist)

        clustered.append(_merge_cluster_properties(page_no, flag, cluster))
    return clustered


def cluster_review_items(
    items: list[ReviewItem], max_dist: float = SPATIAL_CLUSTER_DISTANCE_PX
) -> list[ReviewItem]:
    """Group contiguous low-confidence words on the same page into unified review blocks."""
    if not items:
        return []

    grouped: dict[tuple[int, OcrFlag], list[ReviewItem]] = {}
    non_clusterable: list[ReviewItem] = []

    for item in items:
        if _is_clusterable(item):
            grouped.setdefault((item.page, item.flag), []).append(item)
        else:
            non_clusterable.append(item)

    clustered: list[ReviewItem] = []
    for (page_no, flag), p_items in grouped.items():
        clustered.extend(_build_clusters_for_group(page_no, flag, p_items, max_dist))

    return non_clusterable + clustered


def _process_word_item(
    word: Any, page_no: int, threshold: float
) -> tuple[OcrWord, list[ReviewItem], float | None]:
    content = getattr(word, "content", "") or ""
    span = getattr(word, "span", None)
    poly = getattr(word, "polygon", None)
    offset = getattr(span, "offset", None)
    length = getattr(span, "length", None)
    val, problem = _read_confidence(getattr(word, "confidence", None))

    ocr_word = OcrWord(
        page=page_no,
        content=content,
        confidence=val,
        span_offset=offset if isinstance(offset, int) else None,
        span_length=length if isinstance(length, int) else None,
        polygon=tuple(float(p) for p in poly) if poly else None,
    )

    items: list[ReviewItem] = []
    if problem is not None:
        items.append(
            ReviewItem(
                page=page_no,
                flag=problem,
                confidence=None,
                content=content,
                span_offset=ocr_word.span_offset,
                span_length=ocr_word.span_length,
                polygon=ocr_word.polygon,
            )
        )
        return ocr_word, items, None

    if val is not None and val < threshold:
        items.append(
            ReviewItem(
                page=page_no,
                flag=OcrFlag.LOW_CONFIDENCE,
                confidence=round(val, 4),
                content=content,
                span_offset=ocr_word.span_offset,
                span_length=ocr_word.span_length,
                polygon=ocr_word.polygon,
            )
        )

    return ocr_word, items, val


def _process_page_words(
    page: Any, page_no: int, threshold: float
) -> tuple[list[OcrWord], list[ReviewItem], list[float]]:
    words = list(getattr(page, "words", None) or [])
    if not words:
        has_lines = bool(getattr(page, "lines", None))
        flag = OcrFlag.UNSCORED_PAGE if has_lines else OcrFlag.NO_TEXT
        return [], [ReviewItem(page=page_no, flag=flag, confidence=None)], []

    words_out: list[OcrWord] = []
    raw_items: list[ReviewItem] = []
    scores: list[float] = []

    for word in words:
        w_obj, w_items, val = _process_word_item(word, page_no, threshold)
        words_out.append(w_obj)
        raw_items.extend(w_items)
        if val is not None:
            scores.append(val)

    return words_out, raw_items, scores


def evaluate_ocr_confidence(
    result: Any,
    threshold: float = DEFAULT_OCR_CONFIDENCE_THRESHOLD,
    enable_clustering: bool = True,
) -> OcrConfidenceReport:
    """Assess word-level OCR confidence with optional spatial clustering."""
    threshold = validate_threshold(threshold)
    pages = list(getattr(result, "pages", None) or [])
    if not pages:
        item = ReviewItem(page=0, flag=OcrFlag.NO_PAGES, confidence=None)
        return OcrConfidenceReport(threshold, (), (item,), 0, 0, None, None)

    words_out: list[OcrWord] = []
    raw_items: list[ReviewItem] = []
    scores: list[float] = []

    for index, page in enumerate(pages):
        page_no = getattr(page, "page_number", index + 1)
        p_words, p_items, p_scores = _process_page_words(page, page_no, threshold)
        words_out.extend(p_words)
        raw_items.extend(p_items)
        scores.extend(p_scores)

    final_items = cluster_review_items(raw_items) if enable_clustering else raw_items

    return OcrConfidenceReport(
        threshold=threshold,
        words=tuple(words_out),
        review_items=tuple(final_items),
        page_count=len(pages),
        scored_word_count=len(scores),
        min_confidence=min(scores) if scores else None,
        mean_confidence=(sum(scores) / len(scores)) if scores else None,
    )


def words_in_span(report: OcrConfidenceReport, offset: int, length: int) -> list[OcrWord]:
    end = offset + length
    return [
        w
        for w in report.words
        if w.span_offset is not None
        and w.span_length is not None
        and w.span_offset < end
        and offset < w.span_offset + w.span_length
    ]
