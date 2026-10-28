"""Shared fixtures. All image fixtures are synthetic and contain no patient information.

Real clinical documents must never be added to ``tests/`` (see .gitignore and README).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from types import SimpleNamespace as NS
from typing import Any

import cv2
import numpy as np
import pytest

# Sentinel strings that stand in for PHI in privacy tests. They must never appear in logs.
PHI_SENTINELS = ("ZZPATIENT-SMITHERSON", "ZZDOB-1901-02-03", "ZZDRUG-QUANTRAXOL")


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Iterator[None]:
    """Unit tests must not depend on a developer's shell, .env file, or Azure credentials."""
    for key in list(os.environ):
        if key.startswith(("AZURE_", "ENABLE_LLM", "ENVIRONMENT", "PREPROCESS_", "OCR_", "LLM_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)  # no stray .env in the CWD
    yield


def synthetic_page(
    width: int = 850,
    height: int = 1100,
    ink: int = 0,
    text: str = "Rx Drug-X 0.5 mg PO BID x 7 days",
    dark_border: bool = False,
) -> np.ndarray:
    img = np.full((height, width, 3), 255, np.uint8)
    for y in range(120, height - 80, 60):
        cv2.putText(img, text, (50, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (ink, ink, ink), 2)
    if dark_border:
        img[:, :40] = 30
        img[height - 40 :, :] = 30
    return img


def rotate_same_canvas(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))


def encode(img: np.ndarray, ext: str = ".png") -> bytes:
    ok, buf = cv2.imencode(ext, img)
    assert ok
    return bytes(buf.tobytes())


@pytest.fixture
def page_png() -> bytes:
    return encode(synthetic_page())


def make_word(content: str, confidence: Any, offset: int = 0, length: int | None = None) -> NS:
    return NS(
        content=content,
        confidence=confidence,
        span=NS(offset=offset, length=len(content) if length is None else length),
        polygon=[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    )


def ocr_result_from_text(text: str, confidences: dict[str, float | None] | None = None) -> NS:
    """Build a fake Document Intelligence AnalyzeResult with correct word spans."""
    confidences = confidences or {}
    words, pos = [], 0
    for token in text.split():
        offset = text.index(token, pos)
        pos = offset + len(token)
        words.append(make_word(token, confidences.get(token, 0.99), offset))
    return NS(content=text, pages=[NS(page_number=1, words=words, lines=[NS(content=text)])])


@pytest.fixture
def fake_ocr_result() -> Callable[..., NS]:
    return ocr_result_from_text
