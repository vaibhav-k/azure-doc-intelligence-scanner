from __future__ import annotations

from typing import Any


def _process_page(
    page: Any, threshold: float, review_queue: list[dict[str, Any]]
) -> list[str]:
    """Helper to process individual pages and evaluate word or line confidence."""
    tokens: list[str] = []
    page_number = getattr(page, "page_number", 1)

    words = getattr(page, "words", None)
    if words:
        for word in words:
            content = getattr(word, "content", "")
            confidence = getattr(word, "confidence", 1.0)

            if confidence < threshold:
                review_queue.append(
                    {
                        "page": page_number,
                        "token": content,
                        "confidence": round(confidence, 4),
                        "status": "FLAGGED_FOR_HUMAN_REVIEW",
                    }
                )
                tokens.append(f"[REVIEW:{content}]")
            else:
                tokens.append(content)
        return tokens

    lines = getattr(page, "lines", None)
    if lines:
        for line in lines:
            tokens.append(getattr(line, "content", ""))

    return tokens


def evaluate_ocr_confidence(
    result: Any, threshold: float = 0.85
) -> tuple[str, list[dict[str, Any]]]:
    """
    Scans Azure Document Intelligence pages and words.
    Flags words below the confidence threshold for human review.
    Returns reconstructed text and the human review queue.
    """
    review_queue: list[dict[str, Any]] = []
    pages = getattr(result, "pages", None)
    if not pages:
        return "", review_queue

    tokens: list[str] = []
    for page in pages:
        tokens.extend(_process_page(page, threshold, review_queue))

    return " ".join(tokens), review_queue
