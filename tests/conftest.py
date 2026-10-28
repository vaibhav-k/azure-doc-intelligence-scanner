"""Shared fixtures. All image fixtures are synthetic and contain no patient information.

Real clinical documents must never be added to ``tests/`` (see .gitignore and README).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import cv2
import numpy as np
import pytest

# Sentinel strings that stand in for PHI in privacy tests. They must never appear in logs.
PHI_SENTINELS = ("ZZPATIENT-SMITHERSON", "ZZDOB-1901-02-03", "ZZDRUG-QUANTRAXOL")


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Unit tests must not depend on a developer's shell, .env file, or Azure credentials."""
    for key in list(os.environ):
        if key.startswith(("AZURE_", "ENABLE_LLM", "ENVIRONMENT", "PREPROCESS_", "OCR_", "LLM_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)  # no stray .env in the CWD


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
    words: list[Any] = []
    pos = 0
    for token in text.split():
        offset = text.index(token, pos)
        pos = offset + len(token)
        words.append(make_word(token, confidences.get(token, 0.99), offset))
    return NS(content=text, pages=[NS(page_number=1, words=words, lines=[NS(content=text)])])


@pytest.fixture
def fake_ocr_result() -> Callable[..., NS]:
    return ocr_result_from_text


def make_pdf(pages: int) -> bytes:
    """A real, blank multi-page PDF built with PDFium (no text, no PHI)."""
    import io

    import pypdfium2 as pdfium

    document = pdfium.PdfDocument.new()
    try:
        for _ in range(pages):
            document.new_page(595, 842)
        buffer = io.BytesIO()
        document.save(buffer)
        return buffer.getvalue()
    finally:
        document.close()


def make_tiff(pages: int, tmp_path: Path) -> bytes:
    """A real multi-page TIFF of synthetic pages."""
    path = tmp_path / "pages.tiff"
    assert cv2.imwritemulti(str(path), [synthetic_page() for _ in range(pages)])
    return path.read_bytes()


# A one-page PDF encrypted with a user password ("secret"), text "x". Generated once with
# reportlab's StandardEncryption; stored compressed because .gitignore excludes *.pdf files.
_LOCKED_PDF_B64 = (
    "eNqtVc9rJEUUPngQe9eDIAuCSCGMJLCZrt/dLUOku2c6jtlNxmR245oOWDNdyfY66Z5092ySRfYu6knEiygS8SKKsh78C0QE"
    "BU/iTfwThJUVPfi6Z7Jpd4S92Ifqeu999b33qt6ravTawRJpMqPx7ttvfYg29DjNiktqgFZ0ojNV6AgBAkXpcLKvkwItpGOd"
    "5OkkG+pFgyCM0sENo9UyzIAgCuKGsbxs6CQq1bRm9lSugxQIzBf16KYu4qFCZicZplGc7CFzK07cJI/PFGtqX6OS1NycDIrj"
    "MQh9GMn0BxbgqrliNVc+mCDUHNllPMi8rKNYeekR2gYZI+GIJrWERDYnTdt2bLSDzJ7Kyuys6YoNPc0wR1VmZdhkaull6XBT"
    "F0Blltti9vURpNTdV3vam/392b+LdiBAIEsL2EZYbvYzlVSUpd5Ap6n0AFxLhddSKU2X0whAV3K9liYzdH4a6JTAV4UapXs1"
    "DlHjcCfF9TRDCyGVVsiEWA0Z4yGT1s3nQ2aJ/UUgyLQq4jRpl4ECkNKQWiTEVIYUl3O+FMGIL8YhZtYqDAQIZBesVi9klL0Q"
    "Uia3YAEPMSanjA/xuqqPD9MsytECCJDl/+cdTimaDHXlHoNXLvvgXoBjnk6XC/LGwe3lEZg4BhOptJSSJGSEgSCBCdgIa5aK"
    "PsxKGL8WMmyBI6sBfolPS8RidZRQpTf0sCg9clYmvAVQChN21IJRHvKXILB+XIz0GQaiZzCx6CGwC75Ylch4DD1nBmqU16tC"
    "1lstHhWQnLlZqCRSGaDXUasjXLtNpW15baeN/SBwuPQs1xciEATbpO0HuE2IFIL73At8D7uiw7kMpNNxue1CpfbQEod6hT42"
    "r6BW4HU837VJ4ARUOJ7tMSIs7gBFh0m/05aUcxc7zMFCdlwZWMTptIXvMd+1LEwoEF5FpJaC9a8enVRNZa7GUADbiFX1vFNv"
    "iby21J7PHjrQ3fS7XVu09bBqkWAEBTQTgOmSTvaK68ihJU9eQEHuG09cWynuPdV87+CL7JGTv//48/0fnnz2wpsn5z76/W60"
    "9vqXr945uPryaxfPf3b4yvozJ1891vt1++tvH7/16KC58tPa+XPfvfNjA318+zn16c9b+Ptf7v016D/92+d33A8uYP7NJxDq"
    "zMss6KNM7xoYOQa+/yHYfibQLrqvk3CxVJbkTOfQOR1xnDkdx/M4buE5ncPkgzpCGJnXObW1Rabikc6qHYdbOTseF0hW97vZ"
    "bSNju0VtLLRWwmFK2xEdYI2VdKQWjGHtaLX8cMSO0ag9OHv//eAsLaEohmp44OmBMJLdFIlpSBtpWiA+nW/GtzRsenXoKiuq"
    "UyDMIkaj0VkPjH8A/87rTA=="
)


def locked_pdf() -> bytes:
    import base64
    import zlib

    return zlib.decompress(base64.b64decode(_LOCKED_PDF_B64))
