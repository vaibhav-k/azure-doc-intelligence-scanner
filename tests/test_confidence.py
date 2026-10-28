from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from azure_scanner.confidence import (
    OcrFlag,
    evaluate_ocr_confidence,
    words_in_span,
)
from tests.conftest import PHI_SENTINELS, make_word


def _result(*pages: NS) -> NS:
    return NS(pages=list(pages))


def _page(words=None, lines=None, n=1):  # type: ignore[no-untyped-def]
    return NS(page_number=n, words=words, lines=lines)


@pytest.mark.parametrize("result", [NS(pages=[]), NS(pages=None), NS(), None])
def test_empty_or_incomplete_results_are_flagged(result: object) -> None:
    report = evaluate_ocr_confidence(result)
    assert report.flags == {OcrFlag.NO_PAGES}
    assert report.needs_review
    assert report.min_confidence is None  # no number invented


def test_page_without_words_or_lines_is_no_text() -> None:
    report = evaluate_ocr_confidence(_result(_page()))
    assert report.flags == {OcrFlag.NO_TEXT}


def test_line_only_page_is_unscored_not_passed() -> None:
    report = evaluate_ocr_confidence(_result(_page(lines=[NS(content="Drug-X 5 mg")])))
    assert report.flags == {OcrFlag.UNSCORED_PAGE}
    assert report.scored_word_count == 0


def test_missing_confidence_is_unknown_not_perfect() -> None:
    w = NS(content="5mg", span=NS(offset=0, length=3))  # no confidence attribute at all
    report = evaluate_ocr_confidence(_result(_page([w])))
    assert report.flags == {OcrFlag.MISSING_CONFIDENCE}
    assert report.words[0].confidence is None
    assert report.min_confidence is None


@pytest.mark.parametrize("bad", ["0.9", float("nan"), -0.1, 1.5, True, object()])
def test_invalid_confidence_values_are_flagged(bad: object) -> None:
    report = evaluate_ocr_confidence(_result(_page([make_word("bid", bad)])))
    assert report.flags == {OcrFlag.INVALID_CONFIDENCE}
    assert report.words[0].confidence is None


def test_zero_confidence_is_low_confidence() -> None:
    report = evaluate_ocr_confidence(_result(_page([make_word("x", 0.0)])))
    assert report.flags == {OcrFlag.LOW_CONFIDENCE}
    assert report.min_confidence == 0.0


@pytest.mark.parametrize("conf, flagged", [(0.8499, True), (0.85, False), (0.99, False), (1.0, False)])
def test_threshold_boundary(conf: float, flagged: bool) -> None:
    report = evaluate_ocr_confidence(_result(_page([make_word("bid", conf)])), threshold=0.85)
    assert report.needs_review is flagged


def test_threshold_is_configurable() -> None:
    page = _page([make_word("bid", 0.9)])
    assert not evaluate_ocr_confidence(_result(page), 0.85).needs_review
    assert evaluate_ocr_confidence(_result(page), 0.95).needs_review


@pytest.mark.parametrize("bad", [0, -1, 1.01, "0.8", True])
def test_invalid_threshold_rejected(bad: object) -> None:
    with pytest.raises(ValueError):
        evaluate_ocr_confidence(_result(), bad)  # type: ignore[arg-type]


def test_mixed_confidence_pages() -> None:
    p1 = _page(
        [
            make_word("Drug-X", 0.99, 0),
            make_word("0.5", 0.41, 7),
            make_word("mg", None, 11),
        ],
        n=1,
    )
    p2 = _page(lines=[NS(content="sig")], n=2)
    report = evaluate_ocr_confidence(_result(p1, p2))
    assert report.flag_counts() == {
        "low_confidence": 1,
        "missing_confidence": 1,
        "unscored_page": 1,
    }
    assert report.scored_word_count == 2
    assert report.min_confidence == 0.41
    assert report.page_count == 2


def test_words_and_lines_on_same_page_uses_word_scores() -> None:
    page = _page(words=[make_word("bid", 0.99)], lines=[NS(content="bid")])
    assert not evaluate_ocr_confidence(_result(page)).needs_review


def test_span_lookup_for_field_provenance() -> None:
    p = _page(
        [
            make_word("Drug-X", 0.99, 0),
            make_word("0.5", 0.41, 7),
            make_word("mg", 0.95, 11),
        ]
    )
    report = evaluate_ocr_confidence(_result(p))
    assert [w.confidence for w in words_in_span(report, 7, 6)] == [0.41, 0.95]


def test_repr_never_contains_ocr_text() -> None:
    page = _page([make_word(PHI_SENTINELS[0], 0.1)])
    report = evaluate_ocr_confidence(_result(page))
    assert PHI_SENTINELS[0] not in repr(report)
    assert PHI_SENTINELS[0] not in repr(report.review_items[0])
