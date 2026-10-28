"""Vendor-neutral EMR integration contract (read-only first).

This module defines *interfaces and safety checks only*. It contains no vendor endpoints,
SDK calls or credentials. A concrete adapter (e.g. for Accuro) may be written only after the
vendor has confirmed the relevant capability - see ``integrations/accuro.py``.

Safety rules enforced here
--------------------------
* Patient-document association comes from the EMR reference supplied by the caller, never
  from OCR or LLM output, and the fetched document must match that reference exactly.
* Results travel with their originating ``SourceDocumentRef``; nothing re-derives the patient.
* Write-back is refused unless (a) the review record is APPROVED by a clinician, (b) it belongs
  to the same processing run, and (c) the sink's write capability has been explicitly marked
  VERIFIED. The default sink refuses all writes.
* Identifiers (patient/document IDs) are never logged; only the correlation ID is.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from azure_scanner.errors import PipelineError
from azure_scanner.logging_utils import log_event, new_correlation_id
from azure_scanner.models import SCHEMA_VERSION
from azure_scanner.pipeline import DocumentPipeline, ProcessingResult, ProcessingStatus
from azure_scanner.review import ReviewRecord, ReviewStatus

logger = logging.getLogger(__name__)


class EmrCapability(str, enum.Enum):
    AUTHENTICATE = "authenticate"
    DISCOVER_NEW_DOCUMENTS = "discover_new_documents"  # polling / webhook / change feed
    FETCH_DOCUMENT_CONTENT = "fetch_document_content"
    FETCH_DOCUMENT_METADATA = "fetch_document_metadata"
    REPORT_PROCESSING_STATUS = "report_processing_status"
    WRITE_REVIEWED_RESULT = "write_reviewed_result"


class CapabilityStatus(str, enum.Enum):
    UNVERIFIED = "unverified"  # default: not confirmed by the vendor
    VERIFIED = "verified"  # confirmed in writing by the vendor AND tested in their sandbox
    UNSUPPORTED = "unsupported"


# ---------------------------------------------------------------------------- errors


class EmrError(PipelineError):
    stage = "emr"


class EmrAuthError(EmrError):
    pass


class EmrUnavailableError(EmrError):
    retryable = True


class EmrNotFoundError(EmrError):
    pass


class EmrAssociationError(EmrError):
    """The fetched document does not match the requested patient/document reference."""


class WriteBackNotAuthorizedError(EmrError):
    pass


# ---------------------------------------------------------------------------- data


@dataclass(frozen=True)
class SourceDocumentRef:
    source_system: str
    patient_id: str = field(repr=False)
    document_id: str = field(repr=False)
    version: str | None = field(default=None, repr=False)  # etag / revision if the EMR exposes one


@dataclass(frozen=True)
class SourceDocument:
    ref: SourceDocumentRef
    content: bytes = field(repr=False)
    media_type: str | None = None
    uploaded_at: datetime | None = None


@dataclass(frozen=True)
class EmrProcessingOutcome:
    ref: SourceDocumentRef
    correlation_id: str
    result: ProcessingResult | None  # None when the document could not be fetched
    error_code: str | None = None
    retryable: bool = False


@dataclass(frozen=True)
class DeliveryReceipt:
    idempotency_key: str
    accepted: bool
    external_reference: str | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------- protocols


@runtime_checkable
class EmrDocumentSource(Protocol):
    """Read-only access to documents attached to patient records."""

    system_name: str

    def fetch(self, ref: SourceDocumentRef) -> SourceDocument:
        """Return the document for ``ref`` or raise an ``EmrError`` subclass."""
        ...


@runtime_checkable
class ResultSink(Protocol):
    """Delivery of clinician-approved results. Must be idempotent on ``idempotency_key``."""

    capabilities: Mapping[EmrCapability, CapabilityStatus]

    def deliver(self, ref: SourceDocumentRef, result: ProcessingResult, record: ReviewRecord,
                idempotency_key: str) -> DeliveryReceipt: ...


class DisabledResultSink:
    """Default sink: refuses every write until a verified, authorised mechanism exists."""

    capabilities: Mapping[EmrCapability, CapabilityStatus] = {
        EmrCapability.WRITE_REVIEWED_RESULT: CapabilityStatus.UNVERIFIED
    }

    def deliver(self, ref: SourceDocumentRef, result: ProcessingResult, record: ReviewRecord,
                idempotency_key: str) -> DeliveryReceipt:
        raise WriteBackNotAuthorizedError("write_back_not_verified")


# ---------------------------------------------------------------------------- operations


def _same_document(a: SourceDocumentRef, b: SourceDocumentRef) -> bool:
    return (a.source_system, a.patient_id, a.document_id) == (b.source_system, b.patient_id, b.document_id) and (
        a.version is None or a.version == b.version
    )


def process_emr_document(
    source: EmrDocumentSource,
    ref: SourceDocumentRef,
    pipeline: DocumentPipeline,
    *,
    correlation_id: str | None = None,
) -> EmrProcessingOutcome:
    """Fetch one document read-only, verify its association, and run the pipeline."""
    cid = correlation_id or new_correlation_id()
    try:
        if ref.source_system != source.system_name:
            raise EmrAssociationError("source_system_mismatch")
        document = source.fetch(ref)
        if not _same_document(document.ref, ref):
            raise EmrAssociationError("fetched_document_does_not_match_reference")
    except EmrError as exc:
        log_event(logger, logging.WARNING, "emr_fetch_failed", correlation_id=cid, stage="emr",
                  error_code=exc.code, retryable=exc.retryable)
        return EmrProcessingOutcome(ref, cid, None, exc.code, exc.retryable)

    result = pipeline.process(document.content, correlation_id=cid)
    return EmrProcessingOutcome(ref, cid, result, result.error_code, result.retryable)


def idempotency_key(ref: SourceDocumentRef, result: ProcessingResult) -> str:
    return f"{ref.source_system}:{ref.document_id}:{result.source_sha256}:{SCHEMA_VERSION}"


def deliver_if_approved(
    sink: ResultSink,
    outcome: EmrProcessingOutcome,
    record: ReviewRecord,
) -> DeliveryReceipt:
    """Deliver a result only after clinician approval through a VERIFIED write capability."""
    if sink.capabilities.get(EmrCapability.WRITE_REVIEWED_RESULT) is not CapabilityStatus.VERIFIED:
        raise WriteBackNotAuthorizedError("write_back_not_verified")
    result = outcome.result
    if result is None or result.processing_status is not ProcessingStatus.COMPLETED:
        raise WriteBackNotAuthorizedError("no_completed_result")
    if record.correlation_id != result.correlation_id:
        raise WriteBackNotAuthorizedError("review_record_does_not_match_result")
    if record.status is not ReviewStatus.APPROVED:
        raise WriteBackNotAuthorizedError("not_clinician_approved")
    receipt = sink.deliver(outcome.ref, result, record, idempotency_key(outcome.ref, result))
    log_event(logger, logging.INFO, "result_delivered", correlation_id=result.correlation_id, stage="emr",
              status="accepted" if receipt.accepted else "not_accepted")
    return receipt
