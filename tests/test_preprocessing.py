"""
Geometric-safety and input-validation tests for preprocessing.

These use synthetic images only. Passing them shows the code does not damage page geometry;
it does NOT establish clinical OCR accuracy, which requires a de-identified evaluation set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from azure_scanner.preprocessing import (
    MAX_DESKEW_DEGREES,
    ImageValidationError,
    InputFormat,
    PreprocessMode,
    deskew_image,
    estimate_skew,
    is_probably_blank,
    preprocess_bytes,
    preprocess_image,
    rotate_expand,
    sniff_format,
)
from tests.conftest import (
    encode,
    locked_pdf,
    make_pdf,
    make_tiff,
    rotate_same_canvas,
    synthetic_page,
)


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _decode(b: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img is not None, "image failed to decode"
    return np.asarray(img)


# ---------- input validation ----------


@pytest.mark.parametrize(
    "data, code",
    [
        (b"", "empty_input"),
        (b"hello, not an image", "unsupported_format"),
        (
            b"\xff\xd8\xff" + b"\x00" * 64,
            "corrupt_or_undecodable_image",
        ),  # JPEG magic, garbage body
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "corrupt_or_undecodable_image"),
        (b"GIF89a" + b"\x00" * 64, "unsupported_format"),
    ],
)
def test_invalid_inputs_raise_coded_errors(data: bytes, code: str) -> None:
    with pytest.raises(ImageValidationError) as exc:
        preprocess_bytes(data)
    assert exc.value.code == code
    assert str(exc.value) == code  # message is the code only (no paths, no content)


def test_oversized_input_rejected_before_decoding(page_png: bytes) -> None:
    with pytest.raises(ImageValidationError, match="input_too_large"):
        preprocess_bytes(page_png, max_input_bytes=100)


def test_tiny_image_rejected() -> None:
    with pytest.raises(ImageValidationError, match="image_too_small"):
        preprocess_bytes(encode(np.zeros((20, 400, 3), np.uint8)))


@pytest.mark.parametrize("value", [255, 0, 128])
def test_uniform_blank_pages_rejected(value: int) -> None:
    with pytest.raises(ImageValidationError, match="blank_or_unreadable_page"):
        preprocess_bytes(encode(np.full((600, 500, 3), value, np.uint8)))


def test_faint_handwriting_is_not_treated_as_blank() -> None:
    faint = synthetic_page(ink=210)  # light-grey strokes on white
    assert not is_probably_blank(_gray(faint))
    assert preprocess_bytes(encode(faint)).payload


def test_jpeg_noise_alone_is_blank() -> None:
    rng = np.random.default_rng(0)
    noisy = np.clip(255 - rng.integers(0, 6, (700, 500, 3)), 0, 255).astype(np.uint8)
    with pytest.raises(ImageValidationError, match="blank_or_unreadable_page"):
        preprocess_bytes(encode(noisy, ".jpg"))


@pytest.mark.parametrize(
    "header, fmt",
    [
        (b"\xff\xd8\xff\xe0", InputFormat.JPEG),
        (b"\x89PNG\r\n\x1a\n", InputFormat.PNG),
        (b"BM", InputFormat.BMP),
        (b"II*\x00", InputFormat.TIFF),
        (b"MM\x00*", InputFormat.TIFF),
        (b"%PDF-1.7", InputFormat.PDF),
    ],
)
def test_format_sniffing(header: bytes, fmt: InputFormat) -> None:
    assert sniff_format(header + b"\x00" * 16) is fmt


def test_pdf_is_passed_through_untouched_even_in_enhanced_mode() -> None:
    pdf = make_pdf(1)
    out = preprocess_bytes(pdf, PreprocessMode.ENHANCED)
    assert out.payload == pdf
    assert out.warnings == ("enhancement_skipped_multipage_format",)


# ---------- page counting ----------


@pytest.mark.parametrize("pages", [1, 3])
def test_pdf_pages_are_counted(pages: int) -> None:
    assert preprocess_bytes(make_pdf(pages)).page_count == pages


@pytest.mark.parametrize("pages", [1, 4])
def test_tiff_pages_are_counted(pages: int, tmp_path: Path) -> None:
    assert preprocess_bytes(make_tiff(pages, tmp_path)).page_count == pages


def test_single_image_is_one_page() -> None:
    assert preprocess_bytes(encode(synthetic_page(), ".png")).page_count == 1


@pytest.mark.parametrize(
    "data, code",
    [
        (b"%PDF-1.7\n%not really a pdf\n", "corrupt_or_undecodable_pdf"),
        (
            b"II*\x00\xff\xff\xff\x7f",
            "corrupt_or_undecodable_image",
        ),  # IFD offset past the end
        (
            b"II*\x00\x08\x00\x00\x00\x00\x00\x08\x00\x00\x00",
            "corrupt_or_undecodable_image",
        ),  # cycle
    ],
)
def test_unparseable_multipage_inputs_are_rejected(data: bytes, code: str) -> None:
    with pytest.raises(ImageValidationError, match=code):
        preprocess_bytes(data)


def test_password_protected_pdf_is_rejected() -> None:
    with pytest.raises(ImageValidationError, match="pdf_password_protected"):
        preprocess_bytes(locked_pdf())


def test_page_limit_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    import azure_scanner.preprocessing as prep

    monkeypatch.setattr(prep, "MAX_PAGES", 2)
    with pytest.raises(ImageValidationError, match="too_many_pages"):
        preprocess_bytes(make_pdf(3))


# ---------- default (minimal) mode preserves the original ----------


@pytest.mark.parametrize("ext", [".png", ".jpg"])
def test_minimal_mode_forwards_original_bytes(ext: str) -> None:
    data = encode(synthetic_page(), ext)
    out = preprocess_bytes(data, PreprocessMode.MINIMAL)
    assert out.payload == data
    assert out.transformations == ()
    assert (out.width, out.height) == (850, 1100)


def test_huge_image_is_downscaled_not_rejected() -> None:
    img = np.full((60, 10_050, 3), 255, np.uint8)
    cv2.putText(img, "Drug-X 0.5 mg", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    out = preprocess_bytes(encode(img))
    assert "downscaled" in out.transformations
    decoded = _decode(out.payload)
    assert max(decoded.shape[:2]) <= 10_000


# ---------- deskew safety (regression for the original 90-degree rotate/crop defect) ----------


@pytest.mark.parametrize("width, height", [(850, 1100), (1400, 800)], ids=["portrait", "landscape"])
@pytest.mark.parametrize("skew", [-8, -3, 2, 6])
def test_deskew_estimates_small_skews_on_portrait_and_landscape(width: int, height: int, skew: float) -> None:
    img = rotate_same_canvas(synthetic_page(width, height), skew)
    result = estimate_skew(_gray(img))
    assert result.applied
    assert result.angle is not None
    assert abs(result.angle + skew) <= 0.5


def test_straight_page_is_not_rotated() -> None:
    img = synthetic_page()
    result = estimate_skew(_gray(img))
    assert not result.applied
    assert result.reason == "below_minimum"
    assert deskew_image(img) is img


@pytest.mark.parametrize("skew", [0, 3, -3])
def test_dark_border_does_not_cause_wrong_rotation(skew: float) -> None:
    img = rotate_same_canvas(synthetic_page(dark_border=True), skew)
    result = estimate_skew(_gray(img))
    if result.applied:
        assert result.angle is not None
        assert abs(result.angle + skew) <= 0.5
    else:
        assert skew == 0


def test_skew_beyond_bound_is_refused_not_forced() -> None:
    img = rotate_same_canvas(synthetic_page(), 15)
    result = estimate_skew(_gray(img))
    assert not result.applied
    assert result.reason == "out_of_range"
    assert deskew_image(img) is img


def test_single_diagonal_stroke_cannot_drive_rotation() -> None:
    img = np.full((1100, 850, 3), 255, np.uint8)
    cv2.line(img, (100, 900), (750, 200), (0, 0, 0), 3)  # one long contour at ~47 degrees
    result = estimate_skew(_gray(img))
    assert not result.applied


def test_scattered_marks_without_text_lines_are_not_rotated() -> None:
    rng = np.random.default_rng(42)
    img = np.full((1100, 850, 3), 255, np.uint8)
    for _ in range(400):
        x, y = int(rng.integers(20, 830)), int(rng.integers(20, 1080))
        cv2.circle(img, (x, y), 2, (0, 0, 0), -1)
    result = estimate_skew(_gray(img))
    assert not result.applied or abs(result.angle or 0) <= MAX_DESKEW_DEGREES


@pytest.mark.parametrize("width, height", [(850, 1100), (1400, 800)])
@pytest.mark.parametrize("angle", [-10.0, -4.5, 3.3, 10.0])
def test_rotate_expand_never_crops(width: int, height: int, angle: float) -> None:
    # Dark frame along every edge on a white page: if any edge content were cropped, dark area drops.
    img = np.full((height, width), 255, np.uint8)
    img[:12, :] = img[-12:, :] = img[:, :12] = img[:, -12:] = 0
    dark_before = int((img < 128).sum())
    out = rotate_expand(img, angle)
    assert out.shape[0] >= height
    assert out.shape[1] >= width
    dark_after = int((out < 128).sum())
    assert dark_after >= 0.97 * dark_before  # interpolation tolerance only; no cropped edges


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(skew=st.floats(min_value=-9.0, max_value=9.0))
def test_property_enhanced_mode_never_large_rotation_or_crop(skew: float) -> None:
    src = rotate_same_canvas(synthetic_page(), skew)
    out = preprocess_bytes(encode(src), PreprocessMode.ENHANCED)
    decoded = _decode(out.payload)
    assert decoded.shape[0] >= src.shape[0]
    assert decoded.shape[1] >= src.shape[1]
    assert out.deskew is not None
    if out.deskew.applied:
        assert abs(out.deskew.angle or 0) <= MAX_DESKEW_DEGREES
        residual = estimate_skew(decoded if decoded.ndim == 2 else _gray(decoded))
        assert residual.angle is None or abs(residual.angle) <= 1.0


# ---------- enhanced mode preserves small clinically meaningful marks ----------


def _count_small_blobs(gray: np.ndarray) -> int:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    return int(sum(1 for i in range(1, n) if 4 <= stats[i][4] <= 60))


def test_enhanced_mode_preserves_decimal_points_and_thin_strokes() -> None:
    img = np.full((800, 700, 3), 255, np.uint8)
    for row in range(10):
        y = 60 + row * 70
        cv2.putText(img, "Drug", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 1)
        for col in range(6):
            cv2.circle(img, (220 + col * 70, y), 2, (90, 90, 90), -1)  # faint decimal-point-sized dots
    # uneven illumination gradient, as in a phone photo
    gradient = np.linspace(0, 70, 700, dtype=np.float32)[None, :, None]
    lit = np.clip(img.astype(np.float32) - gradient, 0, 255).astype(np.uint8)
    before = _count_small_blobs(_gray(img))
    out = preprocess_bytes(encode(lit), PreprocessMode.ENHANCED)
    after = _count_small_blobs(_decode(out.payload))
    assert before >= 60
    assert after >= before  # no dot removed by denoising / illumination flattening
    assert out.transformations[-2:] == ("illumination_normalized", "denoised")


def test_path_wrapper_reads_file(tmp_path, page_png: bytes) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "synthetic.png"
    p.write_bytes(page_png)
    assert preprocess_image(str(p))[:8] == b"\x89PNG\r\n\x1a\n"


def test_path_wrapper_missing_file(tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(FileNotFoundError):
        preprocess_image(str(tmp_path / "missing.png"))


# ---------- decompression-bomb guard (dimensions read from the header, before decoding) ----------


def _with_exif(jpeg: bytes) -> bytes:
    payload = b"Exif\x00\x00" + b"\x00" * 64
    app1 = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    return jpeg[:2] + app1 + jpeg[2:]


@pytest.mark.parametrize(
    "build",
    [
        lambda img: encode(img, ".png"),
        lambda img: encode(img, ".bmp"),
        lambda img: encode(img, ".jpg"),
        lambda img: bytes(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_PROGRESSIVE, 1])[1].tobytes()),
        lambda img: _with_exif(encode(img, ".jpg")),
    ],
    ids=["png", "bmp", "jpeg", "jpeg-progressive", "jpeg-exif"],
)
def test_header_dimensions_match_decoded_dimensions(build: Any) -> None:
    from azure_scanner.preprocessing import raster_dimensions, sniff_format

    img = np.full((321, 457, 3), 255, np.uint8)
    data = build(img)
    fmt = sniff_format(data)
    assert fmt is not None
    assert raster_dimensions(data, fmt) == (457, 321)


def test_top_down_bmp_dimensions_are_positive() -> None:
    from azure_scanner.preprocessing import InputFormat, raster_dimensions

    data = bytearray(encode(np.full((40, 60, 3), 255, np.uint8), ".bmp"))
    data[22:26] = (-40).to_bytes(4, "little", signed=True)
    assert raster_dimensions(bytes(data), InputFormat.BMP) == (60, 40)


def test_oversized_image_is_rejected_before_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import azure_scanner.preprocessing as prep

    def must_not_decode(_: bytes) -> None:
        raise AssertionError("decoded an over-budget image")

    monkeypatch.setattr(prep, "MAX_DECODE_PIXELS", 100 * 100)
    monkeypatch.setattr(prep, "decode_raster", must_not_decode)
    with pytest.raises(ImageValidationError, match="image_dimensions_too_large"):
        preprocess_bytes(encode(np.full((101, 100, 3), 255, np.uint8), ".png"))


@pytest.mark.parametrize("cut", [4, 20])
def test_truncated_jpeg_header_is_rejected(cut: int) -> None:
    data = encode(synthetic_page(), ".jpg")[:cut]
    with pytest.raises(ImageValidationError, match="corrupt_or_undecodable_image"):
        preprocess_bytes(data)


# ---------- HEIF (iPhone photos) ----------


def _heif(brand: bytes) -> bytes:
    return b"\x00\x00\x00\x18ftyp" + brand + b"\x00\x00\x00\x00mif1heic" + b"\x00" * 64


@pytest.mark.parametrize("brand", [b"heic", b"heix", b"mif1"])
def test_heif_is_accepted_and_forwarded_untouched(brand: bytes) -> None:
    from azure_scanner.preprocessing import InputFormat

    data = _heif(brand)
    out = preprocess_bytes(data, PreprocessMode.ENHANCED)
    assert out.input_format is InputFormat.HEIF
    assert out.payload == data
    assert out.page_count is None  # cannot be counted locally, so the page check is skipped
    assert out.warnings == ("enhancement_skipped_heif",)


def test_avif_is_not_mistaken_for_heif() -> None:
    with pytest.raises(ImageValidationError, match="unsupported_format"):
        preprocess_bytes(_heif(b"avif"))


@pytest.mark.parametrize(
    "data, expected",
    [
        # SOI, fill byte, RST0 (no length), APP0, then SOF0 with 7x5 pixels
        (
            b"\xff\xd8\xff\xff\xd0\xff\xe0\x00\x04ab\xff\xc0\x00\x11\x08\x00\x05\x00\x07" + b"\x00" * 16,
            (7, 5),
        ),
        (
            b"\xff\xd8\xff\xda\x00\x08" + b"\x00" * 16,
            None,
        ),  # scan data before any frame header
        (b"\xff\xd8\xff\xe0\x00\x01" + b"\x00" * 16, None),  # impossible segment length
        (b"\xff\xd8\x00\x00" + b"\x00" * 16, None),  # not a marker
        (b"\xff\xd8\xff\xc0\x00\x11\x08", None),  # frame header cut short
    ],
)
def test_jpeg_header_parser_handles_unusual_and_malformed_markers(
    data: bytes, expected: tuple[int, int] | None
) -> None:
    from azure_scanner.preprocessing import InputFormat, raster_dimensions

    assert raster_dimensions(data, InputFormat.JPEG) == expected


def test_other_raster_header_variants() -> None:
    from azure_scanner.preprocessing import InputFormat, raster_dimensions

    core = (
        b"BM"
        + b"\x00" * 12
        + (12).to_bytes(4, "little")
        + (9).to_bytes(2, "little")
        + (4).to_bytes(2, "little")
    )
    assert raster_dimensions(core + b"\x00" * 8, InputFormat.BMP) == (
        9,
        4,
    )  # OS/2 core header
    assert raster_dimensions(b"BM" + b"\x00" * 10, InputFormat.BMP) is None  # truncated
    assert raster_dimensions(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16, InputFormat.PNG) is None  # no IHDR
    assert raster_dimensions(b"%PDF-1.7", InputFormat.PDF) is None
