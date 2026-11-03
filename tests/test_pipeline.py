"""End-to-end pipeline tests, fully offline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import openai
import pytest

from azure_scanner.llm_parser import LlmExtractor
from azure_scanner.pipeline import (
    STAGES,
    DocumentPipeline,
    ProcessingResult,
    ProcessingStatus,
)
from azure_scanner.review import ReviewStatus
from azure_scanner.service import DocumentScannerService
from tests.conftest import ocr_result_from_text
from tests.factories import CLEAN_TEXT, clean_med, doc
from tests.test_llm_parser import Recorder, httpx, ok

FIXED = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)


def _di_client(result: Any = None, error: Exception | None = None) -> MagicMock:
    client = MagicMock()
    if error is not None:
        client.begin_analyze_document.side_effect = error
    poller = client.begin_analyze_document.return_value
    poller.result.return_value = result
    poller.done.return_value = True
    return client


def _extractor(recorder: Recorder) -> LlmExtractor:
    client = openai.OpenAI(
        base_url="https://unit-test.invalid/openai/v1/",
        api_key="t",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(recorder)),
    )
    return LlmExtractor(client, "test-deployment")


def _pipeline(
    ocr_result: Any = None,
    recorder: Recorder | None = None,
    di_error: Exception | None = None,
    llm: bool = True,
) -> tuple[DocumentPipeline, MagicMock]:
    di = _di_client(
        ocr_result if ocr_result is not None else ocr_result_from_text(CLEAN_TEXT),
        di_error,
    )
    scanner = DocumentScannerService(di, "prebuilt-layout", timeout_seconds=5)
    extractor = (
        _extractor(recorder or Recorder([ok(doc(medications=[clean_med()]).model_dump_json())]))
        if llm
        else None
    )
    return DocumentPipeline(scanner, extractor, clock=lambda: FIXED), di


@pytest.fixture(autouse=True)
def _no_sdk_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("openai._base_client.time.sleep", lambda s: None)


def test_happy_path_runs_all_stages_in_order_and_needs_review(page_png: bytes) -> None:
    pipe, di = _pipeline()
    result = pipe.process(page_png, correlation_id="cid-1")
    assert result.processing_status is ProcessingStatus.COMPLETED
    assert result.stages_completed == STAGES
    assert result.review.status is ReviewStatus.NEEDS_REVIEW
    assert result.extraction is not None
    assert result.extraction.medications[0].drug_name.verbatim == "Drug-X"
    assert result.error_code is None
    assert result.failed_stage is None
    assert result.llm.prompt_tokens == 120
    assert result.ocr is not None
    assert result.ocr.page_count == 1
    assert di.begin_analyze_document.call_args.kwargs["body"].bytes_source == page_png
    assert len(result.source_sha256) == 64
    assert result.created_at == FIXED


def test_result_serialization_round_trip(page_png: bytes) -> None:
    result = _pipeline()[0].process(page_png)
    again = ProcessingResult.model_validate_json(result.model_dump_json())
    assert again == result


def test_invalid_input_fails_before_any_azure_call() -> None:
    pipe, di = _pipeline()
    result = pipe.process(b"not an image")
    assert result.processing_status is ProcessingStatus.FAILED
    assert result.failed_stage == "input_validation"
    assert result.error_code == "unsupported_format"
    di.begin_analyze_document.assert_not_called()
