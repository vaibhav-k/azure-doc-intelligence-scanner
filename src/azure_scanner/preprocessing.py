"""
Input validation and conservative, opt-in image enhancement.

Design rules
------------
* Preprocessing is a measured intervention, not a default. ``PreprocessMode.MINIMAL`` (the
  default) validates the input and forwards the ORIGINAL bytes to Document Intelligence,
  which performs its own orientation and skew handling.
* ``PreprocessMode.ENHANCED`` applies bounded deskew, illumination flattening and gentle
  denoising. It never binarises, never rotates by large angles, and never crops: rotation
  uses an expanded canvas. Enable it only if your evaluation set shows it helps.
* Every failure raises ``ImageValidationError`` with a stable, PHI-free ``code``.
* Synthetic-image tests of this module verify geometric safety only; they do not establish
  clinical OCR accuracy.
"""

from __future__ import annotations

import enum
import logging
import struct
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from azure_scanner.errors import PipelineError

logger = logging.getLogger(__name__)

Image = NDArray[Any]  # explicit type parameters keep older numpy stubs (CI minimum versions) happy

# Azure Document Intelligence v4.0 documents 50 px minimum and 10,000 px maximum per side
# (verify against your pricing tier). Inputs above the max side are downscaled, not rejected.
MIN_SIDE_PX = 50
MAX_SIDE_PX = 10_000
DEFAULT_MAX_INPUT_BYTES = 50 * 1024 * 1024
# Document Intelligence processes at most 2,000 pages of a PDF/TIFF (the free F0 tier: only 2).
MAX_PAGES = 2_000

MAX_DESKEW_DEGREES = 10.0
MIN_DESKEW_DEGREES = 0.5
_DESKEW_ANALYSIS_SIDE = 1200
_DESKEW_MIN_GAIN = 1.10  # best-angle profile score must beat 0 degrees by >= 10%
_INK_DELTA = 20  # grey levels darker than local background that count as ink
_BLANK_INK_RATIO = 0.0005


class ImageValidationError(PipelineError):
    """Unreadable, unsupported, blank or out-of-range input. ``code`` is PHI-free."""

    stage = "input_validation"


class InputFormat(str, enum.Enum):
    JPEG = "jpeg"
    PNG = "png"
    BMP = "bmp"
    TIFF = "tiff"
    PDF = "pdf"


_RASTER_FORMATS = {InputFormat.JPEG, InputFormat.PNG, InputFormat.BMP}


class PreprocessMode(str, enum.Enum):
    MINIMAL = "minimal"  # validate; forward original bytes (downscale only if over the size limit)
    ENHANCED = "enhanced"  # + bounded deskew, illumination flattening, light denoise


@dataclass(frozen=True)
class DeskewResult:
    angle: float | None  # estimated skew in degrees (None = not measurable)
    applied: bool
    reason: str  # applied | below_minimum | out_of_range | low_confidence | insufficient_ink


@dataclass(frozen=True)
class PreprocessResult:
    payload: bytes
    input_format: InputFormat
    mode: PreprocessMode
    width: int | None
    height: int | None
    transformations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    deskew: DeskewResult | None = field(default=None)
    # Pages the input contains, counted locally. Compared with the OCR result so that pages the
    # service silently skips (free tier, page limits) can never go unnoticed.
    page_count: int | None = None


def sniff_format(data: bytes) -> InputFormat | None:
    """Identify the container from magic bytes (file extensions are not trusted)."""
    if data[:3] == b"\xff\xd8\xff":
        return InputFormat.JPEG
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return InputFormat.PNG
    if data[:2] == b"BM":
        return InputFormat.BMP
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return InputFormat.TIFF
    if data[:5] == b"%PDF-":
        return InputFormat.PDF
    return None


def _tiff_page_count(data: bytes) -> int:
    """Count TIFF pages by walking the IFD chain (classic TIFF; BigTIFF is not sniffed)."""
    order = "<" if data[:2] == b"II" else ">"

    def read(fmt: str, pos: int) -> int:
        size = struct.calcsize(fmt)
        if pos < 0 or pos + size > len(data):
            raise ImageValidationError("corrupt_or_undecodable_image")
        return int(struct.unpack_from(order + fmt, data, pos)[0])

    offset = read("I", 4)
    seen: set[int] = set()
    pages = 0
    while offset:
        if offset in seen:  # cyclic IFD chain
            raise ImageValidationError("corrupt_or_undecodable_image")
        seen.add(offset)
        pages += 1
        if pages > MAX_PAGES:
            raise ImageValidationError("too_many_pages")
        offset = read("I", offset + 2 + 12 * read("H", offset))
    if pages == 0:
        raise ImageValidationError("corrupt_or_undecodable_image")
    return pages


def _pdf_page_count(data: bytes) -> int:
    """Count PDF pages with PDFium (no rendering). Detects corrupt and password-protected files."""
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_raw

    try:
        document = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        # pypdfium2 5.x attaches ``err_code``; 4.x does not, but PDFium's last error is set in both.
        code = getattr(exc, "err_code", None)
        if code is None:
            code = pdfium_raw.FPDF_GetLastError()
        if code == pdfium_raw.FPDF_ERR_PASSWORD:
            raise ImageValidationError("pdf_password_protected") from None
        raise ImageValidationError("corrupt_or_undecodable_pdf") from None
    try:
        pages = len(document)
    finally:
        document.close()
    if pages == 0:
        raise ImageValidationError("corrupt_or_undecodable_pdf")
    if pages > MAX_PAGES:
        raise ImageValidationError("too_many_pages")
    return pages


def count_pages(data: bytes, fmt: InputFormat) -> int:
    """Number of pages in ``data``. Single-image formats are one page."""
    if fmt is InputFormat.PDF:
        return _pdf_page_count(data)
    if fmt is InputFormat.TIFF:
        return _tiff_page_count(data)
    return 1


def decode_raster(data: bytes) -> Image:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError("corrupt_or_undecodable_image")
    return image


def _to_gray(image: Image) -> Image:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _background(gray: Image) -> Image:
    k = max(15, (min(gray.shape[:2]) // 20) | 1)  # odd kernel, ~5% of the short side
    return cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), min(k, 255))


def ink_mask(gray: Image) -> Image:
    """Pixels noticeably darker than their local background (robust to uneven lighting)."""
    diff = np.asarray(cv2.subtract(_background(gray), gray), dtype=np.int16)
    mask: Image = np.greater(diff, _INK_DELTA).astype(np.uint8)
    return mask


def is_probably_blank(gray: Image) -> bool:
    if float(gray.std()) < 1.5:
        return True
    return float(ink_mask(gray).mean()) < _BLANK_INK_RATIO


def _drop_edge_artifacts(mask: Image) -> Image:
    """Remove large components touching the image edge (table edges, scanner shadows, borders)."""
    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    keep = np.ones(n, dtype=bool)
    keep[0] = False
    for i in range(1, n):
        # Unpack stats safely
        stat_row = stats[i]
        x, y, bw, bh, area = (
            int(stat_row[0]),
            int(stat_row[1]),
            int(stat_row[2]),
            int(stat_row[3]),
            int(stat_row[4]),
        )

        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        large = area > 0.005 * h * w or bw > 0.5 * w or bh > 0.5 * h
        if touches and large:
            keep[i] = False

    # Use np.isin or a boolean mask lookup that satisfies NumPy stubs cleanly,
    # or map valid indices directly:
    valid_labels = np.nonzero(keep)[0]
    out = np.isin(labels, valid_labels).astype(np.uint8) * 255  # or binary mask 1/0
    return out


def _profile_score(mask: Image, angle: float) -> float:
    h, w = mask.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0.0,))
    rows = rotated.sum(axis=1).astype(np.float64)
    return float(np.sum(np.diff(rows) ** 2))


def estimate_skew(gray: Image, max_degrees: float = MAX_DESKEW_DEGREES) -> DeskewResult:
    """
    Projection-profile skew estimate with an explicit confidence gate.

    Returns the rotation (degrees, OpenCV convention: positive = counter-clockwise) that best
    aligns text lines horizontally, or a DeskewResult with ``applied=False`` and a reason.
    Independent of OpenCV's minAreaRect angle convention, which changed between 4.x releases.
    """
    scale = min(1.0, _DESKEW_ANALYSIS_SIDE / max(gray.shape[:2]))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    mask = _drop_edge_artifacts(ink_mask(small))
    if int(mask.sum()) < 300:
        return DeskewResult(None, False, "insufficient_ink")

    search = max_degrees + 2.0  # search slightly wider so a boundary optimum is detectable
    coarse = np.arange(-search, search + 1e-9, 0.5)
    scores = np.array([_profile_score(mask, a) for a in coarse])
    best = float(coarse[int(np.argmax(scores))])
    fine = np.arange(best - 0.5, best + 0.5 + 1e-9, 0.1)
    fine_scores = np.array([_profile_score(mask, a) for a in fine])
    best = round(float(fine[int(np.argmax(fine_scores))]), 1)
    best_score = float(fine_scores.max())
    zero_score = _profile_score(mask, 0.0)

    if abs(best) > max_degrees:
        return DeskewResult(best, False, "out_of_range")
    if abs(best) < MIN_DESKEW_DEGREES:
        return DeskewResult(best, False, "below_minimum")
    if zero_score > 0 and best_score / zero_score < _DESKEW_MIN_GAIN:
        return DeskewResult(best, False, "low_confidence")
    return DeskewResult(best, True, "applied")


def rotate_expand(image: Image, angle: float) -> Image:
    """Rotate about the centre on an expanded canvas so no content is cropped."""
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(np.ceil(h * sin + w * cos)), int(np.ceil(h * cos + w * sin))
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    fill = (
        255.0,
        255.0,
        255.0,
    )  # white; only the first value is used for single-channel images
    return cv2.warpAffine(image, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=fill)


def deskew_image(image: Image, max_degrees: float = MAX_DESKEW_DEGREES) -> Image:
    """Backward-compatible name. Returns the input unchanged unless a confident, bounded skew is found."""
    result = estimate_skew(_to_gray(image), max_degrees)
    return rotate_expand(image, result.angle) if result.applied and result.angle is not None else image


def normalize_illumination(gray: Image) -> Image:
    """Flatten uneven lighting by dividing by the estimated background. No thresholding."""
    bg = _background(gray).astype(np.float32) + 1.0
    out = cv2.divide(gray.astype(np.float32), bg, scale=255.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def convert_to_grayscale(image: Image) -> Image:
    """Converts a BGR image to grayscale (kept for backward compatibility)."""
    return _to_gray(image)


def _encode_png(image: Image) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ImageValidationError("encode_failed")
    return bytes(buf.tobytes())


def preprocess_bytes(
    data: bytes,
    mode: PreprocessMode = PreprocessMode.MINIMAL,
    *,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
) -> PreprocessResult:
    """Validate ``data`` and return the payload to send to Document Intelligence."""
    if not data:
        raise ImageValidationError("empty_input")
    if len(data) > max_input_bytes:
        raise ImageValidationError("input_too_large")
    fmt = sniff_format(data)
    if fmt is None:
        raise ImageValidationError("unsupported_format")

    if fmt not in _RASTER_FORMATS:
        # Multi-page containers go to Document Intelligence untouched.
        warnings = ("enhancement_skipped_multipage_format",) if mode is PreprocessMode.ENHANCED else ()
        return PreprocessResult(data, fmt, mode, None, None, (), warnings, page_count=count_pages(data, fmt))

    image = decode_raster(data)
    h, w = image.shape[:2]
    if min(h, w) < MIN_SIDE_PX:
        raise ImageValidationError("image_too_small")
    gray = _to_gray(image)
    if is_probably_blank(gray):
        raise ImageValidationError("blank_or_unreadable_page")

    transformations: list[str] = []
    payload = data
    deskew: DeskewResult | None = None

    if max(h, w) > MAX_SIDE_PX:
        f = MAX_SIDE_PX / max(h, w)
        image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        gray = _to_gray(image)
        transformations.append("downscaled")
        payload = _encode_png(image)

    if mode is PreprocessMode.ENHANCED:
        deskew = estimate_skew(gray)
        out = gray
        if deskew.applied and deskew.angle is not None:
            out = rotate_expand(out, deskew.angle)
            transformations.append("deskewed")
        out = normalize_illumination(out)
        out = cv2.fastNlMeansDenoising(out, h=3, templateWindowSize=7, searchWindowSize=21)
        transformations += ["illumination_normalized", "denoised"]
        payload = _encode_png(out)
        h, w = out.shape[:2]

    return PreprocessResult(payload, fmt, mode, w, h, tuple(transformations), (), deskew, page_count=1)


def preprocess_image(file_path: str, mode: PreprocessMode = PreprocessMode.ENHANCED) -> bytes:
    """Backward-compatible path-based wrapper. Prefer ``preprocess_bytes``."""
    with open(file_path, "rb") as fh:
        return preprocess_bytes(fh.read(), mode).payload
