"""EMR boundary contract tests using in-memory fakes only (no vendor APIs, no credentials)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import openai
import pytest

from azure_scanner.integrations.accuro import ACCURO_CAPABILITIES, VENDOR_VERIFICATION_REQUIRED
from azure_scanner.integrations.emr import (
    CapabilityStatus,
    DeliveryReceipt,
    DisabledResultSink,
    EmrAuthError,
    EmrCapability,
    EmrDocumentSource,
    EmrError,
    EmrNotFoundError,
    EmrUnavailableError,
    ResultSink,
    SourceDocument,
    SourceDocumentRef,
    WriteBackNotAuthorizedError,
    deliver_if_approved,
    process_emr_document,
)
from azure_scanner.models import HumanConfirmation
from azure_scanner.pipeline import ProcessingStatus
from azure_scanner.review import ReviewPolicy, ReviewStatus, approve, required_confirmation_paths
from tests.conftest import PHI_SENTINELS
from tests.test_pipeline import _pipeline

T0 = datetime(2026, 10, 28, tzinfo=timezone.utc)
REF = SourceDocumentRef("fake-emr", patient_id=PHI_SENTINELS[0], document_id="doc-001", version="v1")


@pytest.fixture(autouse=True)
def _no_sdk_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openai._base_client.time, "sleep", lambda s: None)


@dataclass
class FakeSource:
    content: bytes
    system_name: str = "fake-emr"
    error: EmrError | None = None
    override_ref: SourceDocumentRef | None = None
    calls: list[SourceDocumentRef] = field(default_factory=list)

    def fetch(self, ref: SourceDocumentRef) -> SourceDocument:
        self.calls.append(ref)
        if self.error:
            raise self.error
        return SourceDocument(self.override_ref or ref, self.content, "image/png")


@dataclass
class RecordingSink:
    capabilities: dict[EmrCapability, CapabilityStatus]
    delivered: list[str] = field(default_factory=list)

    def deliver(self, ref, result, record, idempotency_key):  # type: ignore[no-untyped-def]
        if idempotency_key in self.delivered:
            return DeliveryReceipt(idempotency_key, accepted=False)
        self.delivered.append(idempotency_key)
        return DeliveryReceipt(idempotency_key, accepted=True)


VERIFIED_SINK = {EmrCapability.WRITE_REVIEWED_RESULT: CapabilityStatus.VERIFIED}


def _approved(outcome):  # type: ignore[no-untyped-def]
    record = outcome.result.review
    confirmations = [HumanConfirmation(path=p, confirmed_value="ok", reviewer_id="dr-1", confirmed_at=T0)
                     for p in required_confirmation_paths(record, ReviewPolicy())]
    return approve(record, reviewer_id="dr-1", confirmations=confirmations, acknowledged_reasons=record.reasons)


def test_fakes_satisfy_protocols(page_png: bytes) -> None:
    assert isinstance(FakeSource(page_png), EmrDocumentSource)
    assert isinstance(RecordingSink(VERIFIED_SINK), ResultSink)
    assert isinstance(DisabledResultSink(), ResultSink)


def test_successful_read_only_processing_keeps_reference(page_png: bytes) -> None:
    source = FakeSource(page_png)
    outcome = process_emr_document(source, REF, _pipeline()[0], correlation_id="c1")
    assert outcome.result is not None and outcome.result.processing_status is ProcessingStatus.COMPLETED
    assert outcome.ref is REF and source.calls == [REF]
    assert outcome.result.review.status is ReviewStatus.NEEDS_REVIEW


@pytest.mark.parametrize(
    "error, retryable",
    [(EmrAuthError("emr_auth_failed"), False), (EmrNotFoundError("emr_document_not_found"), False),
     (EmrUnavailableError("emr_unavailable"), True)],
)
def test_fetch_failures_are_reported_without_processing(page_png: bytes, error: EmrError, retryable: bool) -> None:
    pipe, di = _pipeline()
    outcome = process_emr_document(FakeSource(page_png, error=error), REF, pipe)
    assert outcome.result is None and outcome.error_code == error.code and outcome.retryable is retryable
    di.begin_analyze_document.assert_not_called()


@pytest.mark.parametrize(
    "other",
    [SourceDocumentRef("fake-emr", "another-patient", "doc-001", "v1"),
     SourceDocumentRef("fake-emr", PHI_SENTINELS[0], "doc-999", "v1"),
     SourceDocumentRef("fake-emr", PHI_SENTINELS[0], "doc-001", "v2")],
    ids=["wrong-patient", "wrong-document", "stale-version"],
)
def test_cross_patient_or_stale_document_is_never_processed(page_png: bytes, other: SourceDocumentRef) -> None:
    pipe, di = _pipeline()
    outcome = process_emr_document(FakeSource(page_png, override_ref=other), REF, pipe)
    assert outcome.result is None and outcome.error_code == "fetched_document_does_not_match_reference"
    di.begin_analyze_document.assert_not_called()


def test_source_system_mismatch(page_png: bytes) -> None:
    outcome = process_emr_document(FakeSource(page_png, system_name="other"), REF, _pipeline()[0])
    assert outcome.error_code == "source_system_mismatch"


def test_default_sink_refuses_even_approved_results(page_png: bytes) -> None:
    outcome = process_emr_document(FakeSource(page_png), REF, _pipeline()[0])
    with pytest.raises(WriteBackNotAuthorizedError, match="write_back_not_verified"):
        deliver_if_approved(DisabledResultSink(), outcome, _approved(outcome))


def test_unverified_capability_refuses_write(page_png: bytes) -> None:
    outcome = process_emr_document(FakeSource(page_png), REF, _pipeline()[0])
    sink = RecordingSink({EmrCapability.WRITE_REVIEWED_RESULT: CapabilityStatus.UNVERIFIED})
    with pytest.raises(WriteBackNotAuthorizedError):
        deliver_if_approved(sink, outcome, _approved(outcome))
    assert sink.delivered == []


def test_unapproved_result_is_never_written(page_png: bytes) -> None:
    outcome = process_emr_document(FakeSource(page_png), REF, _pipeline()[0])
    sink = RecordingSink(VERIFIED_SINK)
    assert outcome.result is not None
    with pytest.raises(WriteBackNotAuthorizedError, match="not_clinician_approved"):
        deliver_if_approved(sink, outcome, outcome.result.review)
    assert sink.delivered == []


def test_review_record_from_another_document_is_rejected(page_png: bytes) -> None:
    pipe = _pipeline()[0]
    a = process_emr_document(FakeSource(page_png), REF, pipe, correlation_id="a")
    b = process_emr_document(FakeSource(page_png), REF, pipe, correlation_id="b")
    with pytest.raises(WriteBackNotAuthorizedError, match="does_not_match"):
        deliver_if_approved(RecordingSink(VERIFIED_SINK), a, _approved(b))


def test_failed_processing_is_never_written(page_png: bytes) -> None:
    outcome = process_emr_document(FakeSource(b"garbage"), REF, _pipeline()[0])
    assert outcome.result is not None and outcome.result.processing_status is ProcessingStatus.FAILED
    with pytest.raises(WriteBackNotAuthorizedError, match="no_completed_result"):
        deliver_if_approved(RecordingSink(VERIFIED_SINK), outcome, outcome.result.review)


def test_verified_sink_and_approval_deliver_once_idempotently(page_png: bytes) -> None:
    outcome = process_emr_document(FakeSource(page_png), REF, _pipeline()[0])
    sink, record = RecordingSink(VERIFIED_SINK), _approved(outcome)
    first = deliver_if_approved(sink, outcome, record)
    second = deliver_if_approved(sink, outcome, record)
    assert first.accepted and not second.accepted and len(sink.delivered) == 1
    assert first.idempotency_key.startswith("fake-emr:doc-001:")


def test_identifiers_never_logged(page_png: bytes, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    process_emr_document(FakeSource(page_png, error=EmrNotFoundError("emr_document_not_found")), REF, _pipeline()[0])
    process_emr_document(FakeSource(page_png), REF, _pipeline()[0])
    logged = " ".join(r.getMessage() + repr(getattr(r, "safe_fields", "")) for r in caplog.records)
    assert PHI_SENTINELS[0] not in logged and "doc-001" not in logged
    assert PHI_SENTINELS[0] not in repr(REF)


def test_accuro_capabilities_are_all_unverified() -> None:
    assert set(ACCURO_CAPABILITIES) == set(EmrCapability)
    assert set(ACCURO_CAPABILITIES.values()) == {CapabilityStatus.UNVERIFIED}
    assert len(VENDOR_VERIFICATION_REQUIRED) >= 8


def test_core_pipeline_has_no_vendor_imports() -> None:
    import pathlib

    import azure_scanner

    root = pathlib.Path(azure_scanner.__file__).parent
    for py in root.glob("*.py"):
        assert "integrations" not in py.read_text(), py.name
