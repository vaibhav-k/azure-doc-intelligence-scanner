"""Input validation and conservative, opt-in image enhancement.

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
from dataclasses import dataclass, field

import cv2
import numpy as np

from azure_scanner.errors import PipelineError

logger = logging.getLogger(__name__)

# Azure Document Intelligence v4.0 documents 50 px minimum and 10,000 px maximum per side
# (verify against your pricing tier). Inputs above the max side are downscaled, not rejected.
MIN_SIDE_PX = 50
MAX_SIDE_PX = 10_000
DEFAULT_MAX_INPUT_BYTES = 50 * 1024 * 1024

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


def decode_raster(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError("corrupt_or_undecodable_image")
    return image


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _background(gray: np.ndarray) -> np.ndarray:
    k = max(15, (min(gray.shape[:2]) // 20) | 1)  # odd kernel, ~5% of the short side
    return cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), min(k, 255))


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Pixels noticeably darker than their local background (robust to uneven lighting)."""
    diff = cv2.subtract(_background(gray), gray)
    return (diff > _INK_DELTA).astype(np.uint8)


def is_probably_blank(gray: np.ndarray) -> bool:
    if float(gray.std()) < 1.5:
        return True
    return float(ink_mask(gray).mean()) < _BLANK_INK_RATIO


def _drop_edge_artifacts(mask: np.ndarray) -> np.ndarray:
    """Remove large components touching the image edge (table edges, scanner shadows, borders)."""
    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.ones(n, dtype=bool)
    keep[0] = False
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        large = area > 0.005 * h * w or bw > 0.5 * w or bh > 0.5 * h
        if touches and large:
            keep[i] = False
    return keep[labels].astype(np.uint8)


def _profile_score(mask: np.ndarray, angle: float) -> float:
    h, w = mask.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    rows = rotated.sum(axis=1).astype(np.float64)
    return float(np.sum(np.diff(rows) ** 2))


def estimate_skew(gray: np.ndarray, max_degrees: float = MAX_DESKEW_DEGREES) -> DeskewResult:
    """Projection-profile skew estimate with an explicit confidence gate.

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
    best = float(coarse[int(scores.argmax())])
    fine = np.arange(best - 0.5, best + 0.5 + 1e-9, 0.1)
    fine_scores = np.array([_profile_score(mask, a) for a in fine])
    best = round(float(fine[int(fine_scores.argmax())]), 1)
    best_score = float(fine_scores.max())
    zero_score = _profile_score(mask, 0.0)

    if abs(best) > max_degrees:
        return DeskewResult(best, False, "out_of_range")
    if abs(best) < MIN_DESKEW_DEGREES:
        return DeskewResult(best, False, "below_minimum")
    if zero_score > 0 and best_score / zero_score < _DESKEW_MIN_GAIN:
        return DeskewResult(best, False, "low_confidence")
    return DeskewResult(best, True, "applied")


def rotate_expand(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the centre on an expanded canvas so no content is cropped."""
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(np.ceil(h * sin + w * cos)), int(np.ceil(h * cos + w * sin))
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    fill: tuple[int, int, int] | int = (255, 255, 255) if image.ndim == 3 else 255
    return cv2.warpAffine(image, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=fill)


def deskew_image(image: np.ndarray, max_degrees: float = MAX_DESKEW_DEGREES) -> np.ndarray:
    """Backward-compatible name. Returns the input unchanged unless a confident, bounded skew is found."""
    result = estimate_skew(_to_gray(image), max_degrees)
    return rotate_expand(image, result.angle) if result.applied and result.angle is not None else image


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Flatten uneven lighting by dividing by the estimated background. No thresholding."""
    bg = _background(gray).astype(np.float32) + 1.0
    out = cv2.divide(gray.astype(np.float32), bg, scale=255.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts a BGR image to grayscale (kept for backward compatibility)."""
    return _to_gray(image)


def _encode_png(image: np.ndarray) -> bytes:
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
        return PreprocessResult(data, fmt, mode, None, None, (), warnings)

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

    return PreprocessResult(payload, fmt, mode, w, h, tuple(transformations), (), deskew)


def preprocess_image(file_path: str, mode: PreprocessMode = PreprocessMode.ENHANCED) -> bytes:
    """Backward-compatible path-based wrapper. Prefer ``preprocess_bytes``."""
    with open(file_path, "rb") as fh:
        return preprocess_bytes(fh.read(), mode).payload
