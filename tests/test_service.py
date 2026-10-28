from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ServiceRequestError,
)

from azure_scanner.config import Settings
from azure_scanner.errors import OcrError
from azure_scanner.preprocessing import ImageValidationError, PreprocessMode
from azure_scanner.service import DocumentScannerService, build_di_client
from tests.conftest import PHI_SENTINELS


def _client(result: object = None, done: bool = True) -> MagicMock:
    client = MagicMock()
    poller = client.begin_analyze_document.return_value
    poller.result.return_value = result if result is not None else MagicMock(pages=[1, 2])
    poller.done.return_value = done
    return client


def _svc(client: MagicMock, model: str = "prebuilt-layout", **kw: object) -> DocumentScannerService:
    return DocumentScannerService(client, model, timeout_seconds=5, **kw)  # type: ignore[arg-type]


def test_success_sends_original_bytes_in_minimal_mode(page_png: bytes) -> None:
    client = _client()
    result = _svc(client).analyze_bytes(page_png, correlation_id="c1")
    kwargs = client.begin_analyze_document.call_args.kwargs
    assert kwargs["body"].bytes_source == page_png
    assert kwargs["model_id"] == "prebuilt-layout"
    assert "output_content_format" in kwargs
    client.begin_analyze_document.return_value.result.assert_called_once_with(timeout=5)
    assert result is client.begin_analyze_document.return_value.result.return_value


def test_enhanced_mode_sends_processed_png(page_png: bytes) -> None:
    client = _client()
    _svc(client, preprocess_mode=PreprocessMode.ENHANCED).analyze_bytes(page_png, correlation_id="c1")
    sent = client.begin_analyze_document.call_args.kwargs["body"].bytes_source
    assert sent != page_png
    assert sent[:8] == b"\x89PNG\r\n\x1a\n"


def test_markdown_only_requested_for_layout(page_png: bytes) -> None:
    client = _client()
    _svc(client, "prebuilt-read").analyze_bytes(page_png, correlation_id="c1")
    assert "output_content_format" not in client.begin_analyze_document.call_args.kwargs


def test_unfinished_poller_is_a_timeout_not_a_result(page_png: bytes) -> None:
    """azure-core's LROPoller.result(timeout) returns silently on timeout; we must detect it."""
    client = _client(done=False)
    with pytest.raises(OcrError) as exc:
        _svc(client).analyze_bytes(page_png, correlation_id="c1")
    assert exc.value.code == "ocr_timeout"
    assert exc.value.retryable


@pytest.mark.parametrize(
    "error, code, retryable",
    [
        (ClientAuthenticationError(message="401"), "ocr_auth_failed", False),
        (HttpResponseError(message="bad"), "ocr_http_error", False),
        (ServiceRequestError(message="dns"), "ocr_connection_error", True),
    ],
)
def test_sdk_errors_map_to_coded_errors(
    page_png: bytes, error: Exception, code: str, retryable: bool
) -> None:
    client = MagicMock()
    client.begin_analyze_document.side_effect = error
    with pytest.raises(OcrError) as exc:
        _svc(client).analyze_bytes(page_png, correlation_id="c1")
    assert (exc.value.code, exc.value.retryable) == (code, retryable)


@pytest.mark.parametrize("status, retryable", [(429, True), (503, True), (400, False), (404, False)])
def test_http_status_retryability(page_png: bytes, status: int, retryable: bool) -> None:
    err = HttpResponseError(message="x")
    err.status_code = status
    client = MagicMock()
    client.begin_analyze_document.return_value.result.side_effect = err
    with pytest.raises(OcrError) as exc:
        _svc(client).analyze_bytes(page_png, correlation_id="c1")
    assert exc.value.code == f"ocr_http_{status}"
    assert exc.value.retryable is retryable


def test_none_result_is_an_error(page_png: bytes) -> None:
    client = _client()
    client.begin_analyze_document.return_value.result.return_value = None
    with pytest.raises(OcrError, match="ocr_empty_result"):
        _svc(client).analyze_bytes(page_png, correlation_id="c1")


def test_invalid_image_never_reaches_azure() -> None:
    client = _client()
    with pytest.raises(ImageValidationError):
        _svc(client).analyze_bytes(b"junk", correlation_id="c1")
    client.begin_analyze_document.assert_not_called()


def test_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        _svc(_client()).analyze_document("nonexistent_note.jpg")


def test_constructor_validates_timeout() -> None:
    with pytest.raises(ValueError):
        DocumentScannerService(MagicMock(), timeout_seconds=0)


def test_logs_contain_no_paths_or_sdk_messages(
    page_png: bytes, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:  # type: ignore[no-untyped-def]
    caplog.set_level(logging.DEBUG)
    path: Path = tmp_path / f"{PHI_SENTINELS[0]}_scan.png"
    path.write_bytes(page_png)
    client = MagicMock()
    client.begin_analyze_document.side_effect = HttpResponseError(message=f"echo {PHI_SENTINELS[2]}")
    with pytest.raises(OcrError) as exc:
        _svc(client).analyze_document(str(path), correlation_id="c1")
    assert PHI_SENTINELS[2] not in str(exc.value)
    for record in caplog.records:
        text = record.getMessage() + repr(getattr(record, "safe_fields", {}))
        assert PHI_SENTINELS[0] not in text and PHI_SENTINELS[2] not in text


@pytest.mark.parametrize("with_key", [True, False])
def test_build_client_offline(monkeypatch: pytest.MonkeyPatch, with_key: bool) -> None:
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://x.cognitiveservices.azure.com/")
    if with_key:
        monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "k")
    client = build_di_client(Settings())  # type: ignore[call-arg]
    assert type(client).__name__ == "DocumentIntelligenceClient"  # constructed without network calls
