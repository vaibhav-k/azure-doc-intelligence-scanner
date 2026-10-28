"""End-to-end pipeline tests, fully offline.

Real ``DocumentScannerService`` + mocked Document Intelligence client; real ``LlmExtractor``
+ real OpenAI SDK over an in-memory transport. Only network I/O is faked.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import openai
import pytest
from azure.core.exceptions import ServiceRequestError
from pydantic import ValidationError

from azure_scanner.llm_parser import LOW_CONF_OPEN, LlmExtractor
from azure_scanner.models import ClinicalDocumentExtraction
from azure_scanner.pipeline import STAGES, DocumentPipeline, ProcessingResult, ProcessingStatus
from azure_scanner.review import ReviewStatus, new_record
from azure_scanner.service import DocumentScannerService
from tests.conftest import PHI_SENTINELS, ocr_result_from_text
from tests.factories import CLEAN_TEXT, clean_med, doc, fld
from tests.test_llm_parser import Recorder, httpx, ok, status

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
    client = openai.OpenAI(base_url="https://unit-test.invalid/openai/v1/", api_key="t", max_retries=0,
                           http_client=httpx.Client(transport=httpx.MockTransport(recorder)))
    return LlmExtractor(client, "test-deployment")


def _pipeline(ocr_result: Any = None, recorder: Recorder | None = None, di_error: Exception | None = None,
              llm: bool = True) -> tuple[DocumentPipeline, MagicMock]:
    di = _di_client(ocr_result if ocr_result is not None else ocr_result_from_text(CLEAN_TEXT), di_error)
    scanner = DocumentScannerService(di, "prebuilt-layout", timeout_seconds=5)
    extractor = _extractor(recorder or Recorder([ok(doc(medications=[clean_med()]).model_dump_json())])) if llm else None
    return DocumentPipeline(scanner, extractor, clock=lambda: FIXED), di


@pytest.fixture(autouse=True)
def _no_sdk_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openai._base_client.time, "sleep", lambda s: None)


# ------------------------------------------------------------------ success path


def test_happy_path_runs_all_stages_in_order_and_needs_review(page_png: bytes) -> None:
    pipe, di = _pipeline()
    result = pipe.process(page_png, correlation_id="cid-1")
    assert result.processing_status is ProcessingStatus.COMPLETED
    assert result.stages_completed == STAGES
    assert result.review.status is ReviewStatus.NEEDS_REVIEW  # never auto-approved
    assert result.extraction is not None and result.extraction.medications[0].drug_name.verbatim == "Drug-X"
    assert result.error_code is None and result.failed_stage is None
    assert result.llm.prompt_tokens == 120 and result.ocr is not None and result.ocr.page_count == 1
    assert di.begin_analyze_document.call_args.kwargs["body"].bytes_source == page_png
    assert len(result.source_sha256) == 64 and result.created_at == FIXED


def test_low_confidence_ocr_spans_reach_the_llm_marked(page_png: bytes) -> None:
    rec = Recorder([ok(doc(medications=[clean_med()]).model_dump_json())])
    pipe, _ = _pipeline(ocr_result_from_text(CLEAN_TEXT, {"2": 0.4, "BID": None}), rec)
    result = pipe.process(page_png)
    sent = json.loads(rec.requests[0].content)["messages"][1]["content"]
    assert f"{LOW_CONF_OPEN}2]]" in sent and f"{LOW_CONF_OPEN}BID]]" in sent
    assert result.review.priority is not None and result.review.priority.value == "priority"
    assert {i.flag for i in result.ocr_review_items} == {"low_confidence", "missing_confidence"}


def test_llm_disabled_completes_with_manual_transcription_required(page_png: bytes) -> None:
    pipe, _ = _pipeline(llm=False)
    result = pipe.process(page_png)
    assert result.processing_status is ProcessingStatus.COMPLETED
    assert result.extraction is None and "llm_extraction" not in result.stages_completed
    assert "llm_disabled_manual_transcription_required" in result.review.reasons


def test_result_serialization_round_trip(page_png: bytes) -> None:
    result = _pipeline()[0].process(page_png)
    again = ProcessingResult.model_validate_json(result.model_dump_json())
    assert again == result


# ------------------------------------------------------------------ failure propagation


def _assert_failed(result: ProcessingResult, stage: str, code: str) -> None:
    assert result.processing_status is ProcessingStatus.FAILED
    assert (result.failed_stage, result.error_code) == (stage, code)
    assert result.review.status is ReviewStatus.FAILED
    assert result.extraction is None
    assert result.review.events[-1].reason_codes == (code,)


def test_invalid_input_fails_before_any_azure_call() -> None:
    pipe, di = _pipeline()
    result = pipe.process(b"not an image")
    _assert_failed(result, "input_validation", "unsupported_format")
    di.begin_analyze_document.assert_not_called()


def test_ocr_transport_failure_fails_and_skips_llm(page_png: bytes) -> None:
    rec = Recorder([ok()])
    pipe, _ = _pipeline(recorder=rec, di_error=ServiceRequestError(message="dns"))
    result = pipe.process(page_png)
    _assert_failed(result, "ocr", "ocr_connection_error")
    assert result.retryable and rec.requests == []


def test_ocr_timeout_is_failed_not_completed(page_png: bytes) -> None:
    pipe, di = _pipeline()
    di.begin_analyze_document.return_value.done.return_value = False
    _assert_failed(pipe.process(page_png), "ocr", "ocr_timeout")


def test_ocr_with_no_text_fails_and_skips_llm(page_png: bytes) -> None:
    from types import SimpleNamespace as NS

    rec = Recorder([ok()])
    pipe, _ = _pipeline(NS(content="", pages=[]), rec)
    _assert_failed(pipe.process(page_png), "ocr", "ocr_no_text_detected")
    assert rec.requests == []


@pytest.mark.parametrize(
    "response, code",
    [
        (ok(None, refusal="cannot comply"), "model_refused"),
        (ok('{"truncated": ', finish="length"), "output_truncated"),
        (ok('{"wrong": "shape"}'), "schema_validation_failed"),
        (status(401), "llm_auth_failed"),
        (status(429), "llm_rate_limited"),
    ],
)
def test_llm_failures_never_produce_a_completed_result(page_png: bytes, response: Any, code: str) -> None:
    pipe, _ = _pipeline(recorder=Recorder([response]))
    result = pipe.process(page_png)
    _assert_failed(result, "llm_extraction", code)
    assert "llm_extraction" not in result.stages_completed
    assert result.ocr is not None  # earlier-stage evidence is kept for diagnosis


def test_unexpected_bug_becomes_coded_failure(page_png: bytes) -> None:
    pipe, _ = _pipeline(di_error=RuntimeError(f"bug mentioning {PHI_SENTINELS[0]}"))
    result = pipe.process(page_png)
    _assert_failed(result, "ocr", "internal_error")
    assert PHI_SENTINELS[0] not in result.model_dump_json()


def test_result_invariants_cannot_be_bypassed() -> None:
    base: dict[str, Any] = {
        "pipeline_version": "x", "correlation_id": "c", "source_sha256": "0" * 64, "created_at": FIXED,
        "stages_completed": (), "llm": {"enabled": False}, "timings_ms": {},
    }
    with pytest.raises(ValidationError, match="completed_result_must"):
        ProcessingResult(**base, processing_status=ProcessingStatus.COMPLETED, review=new_record("c"))
    with pytest.raises(ValidationError, match="failed_result_must"):
        ProcessingResult(**base, processing_status=ProcessingStatus.FAILED, error_code="x",
                         review=new_record("c"), extraction=ClinicalDocumentExtraction.model_validate(
                             json.loads(doc().model_dump_json())))


# ------------------------------------------------------------------ privacy


def _phi_ocr() -> Any:
    return ocr_result_from_text(f"Patient {PHI_SENTINELS[0]} DOB {PHI_SENTINELS[1]} Rx {PHI_SENTINELS[2]} 2 mg BID",
                                {PHI_SENTINELS[2]: 0.3})


def _phi_extraction_json() -> str:
    m = clean_med().model_copy(update={"drug_name": fld(PHI_SENTINELS[2])})
    return doc(patient_name=fld(PHI_SENTINELS[0]), patient_dob_or_age=fld(PHI_SENTINELS[1]),
               medications=[m]).model_dump_json()


@pytest.mark.parametrize(
    "response", [ok(_phi_extraction_json()), ok("{bad json " + PHI_SENTINELS[0]), status(500)],
    ids=["success", "schema-failure", "server-error"],
)
def test_logs_and_safe_summary_never_contain_phi(page_png: bytes, caplog: pytest.LogCaptureFixture,
                                                 response: Any) -> None:
    caplog.set_level(logging.DEBUG)
    pipe, _ = _pipeline(_phi_ocr(), Recorder([response]))
    result = pipe.process(page_png)
    logged = " ".join(r.getMessage() + repr(getattr(r, "safe_fields", "")) for r in caplog.records)
    summary = json.dumps(result.safe_summary())
    for s in PHI_SENTINELS:
        assert s not in logged, s
        assert s not in summary, s
