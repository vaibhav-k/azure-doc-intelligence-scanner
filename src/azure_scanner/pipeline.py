"""End-to-end orchestration for one document.

Stages (in order): input validation -> preprocessing -> OCR -> confidence assessment ->
structured extraction -> schema & cross-field validation -> human-review routing -> result.

Contract
--------
* ``DocumentPipeline.process`` never raises for document-level problems; it returns a
  ``ProcessingResult``. Invariants (enforced by a validator, so they cannot drift):
    - ``completed`` => review status ``needs_review`` and no error code;
    - ``failed``    => review status ``failed``, an error code, and NO extraction.
  A failed extraction therefore can never look like a completed one.
* Dependencies (OCR service, LLM extractor, clock) are injected; tests use fakes.
* Everything stays in memory. No temporary files or intermediate artefacts are written;
  persisting the result (which contains PHI) is the caller's responsibility.
"""

from __future__ import annotations

import enum
import hashlib
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from azure_scanner import __version__
from azure_scanner.confidence import (
    OcrConfidenceReport,
    OcrFlag,
    evaluate_ocr_confidence,
)
from azure_scanner.errors import OcrError, PipelineError
from azure_scanner.llm_parser import LlmExtractor, annotate_low_confidence
from azure_scanner.logging_utils import log_event, new_correlation_id
from azure_scanner.models import SCHEMA_VERSION, ClinicalDocumentExtraction
from azure_scanner.review import (
    ReviewPolicy,
    ReviewRecord,
    ReviewStatus,
    mark_failed,
    new_record,
    route_for_review,
)
from azure_scanner.service import DocumentScannerService

logger = logging.getLogger(__name__)

STAGES = (
    "input_validation",
    "preprocessing",
    "ocr",
    "confidence_assessment",
    "llm_extraction",
    "validation",
    "review_routing",
)


class ProcessingStatus(str, enum.Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class OcrSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    model_id: str
    page_count: int
    scored_word_count: int
    min_confidence: float | None
    mean_confidence: float | None
    threshold: float
    flag_counts: dict[str, int]


class OcrReviewItemOut(BaseModel):
    """Low-confidence OCR evidence for the reviewer. ``text`` is PHI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page: int
    flag: str
    confidence: float | None
    text: str
    polygon: tuple[float, ...] | None


class LlmSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    model: str | None = None
    prompt_version: str | None = None
    schema_version: str = SCHEMA_VERSION
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ProcessingResult(BaseModel):
    """Full result. Contains PHI (extraction, OCR evidence) - store under access control."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pipeline_version: str
    correlation_id: str
    source_sha256: str
    created_at: datetime
    processing_status: ProcessingStatus
    stages_completed: tuple[str, ...]
    failed_stage: str | None = None
    error_code: str | None = None
    retryable: bool = False
    input_format: str | None = None
    preprocessing: dict[str, Any] | None = None
    ocr: OcrSummary | None = None
    ocr_review_items: tuple[OcrReviewItemOut, ...] = ()
    llm: LlmSummary
    extraction: ClinicalDocumentExtraction | None = None
    review: ReviewRecord
    timings_ms: dict[str, int]

    @model_validator(mode="after")
    def _invariants(self) -> ProcessingResult:
        if self.processing_status is ProcessingStatus.COMPLETED:
            if self.error_code is not None or self.review.status is not ReviewStatus.NEEDS_REVIEW:
                raise ValueError("completed_result_must_need_review_and_have_no_error")
        else:
            if (
                not self.error_code
                or self.review.status is not ReviewStatus.FAILED
                or self.extraction is not None
            ):
                raise ValueError("failed_result_must_have_error_no_extraction_and_failed_review")
        return self

    def safe_summary(self) -> dict[str, Any]:
        """PHI-free view for logs, stdout and monitoring."""
        return {
            "correlation_id": self.correlation_id,
            "processing_status": self.processing_status.value,
            "review_status": self.review.status.value,
            "priority": self.review.priority.value if self.review.priority else None,
            "reason_codes": list(self.review.reasons),
            "failed_stage": self.failed_stage,
            "error_code": self.error_code,
            "retryable": self.retryable,
            "page_count": self.ocr.page_count if self.ocr else None,
            "medication_count": (len(self.extraction.medications) if self.extraction else None),
            "timings_ms": self.timings_ms,
        }


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


_SPAN_FLAGS = {
    OcrFlag.LOW_CONFIDENCE,
    OcrFlag.MISSING_CONFIDENCE,
    OcrFlag.INVALID_CONFIDENCE,
}


def _uncertain_spans(report: OcrConfidenceReport) -> list[tuple[int, int]]:
    return [
        (i.span_offset, i.span_length)
        for i in report.review_items
        if i.flag in _SPAN_FLAGS and i.span_offset is not None and i.span_length
    ]


class DocumentPipeline:
    def __init__(
        self,
        scanner: DocumentScannerService,
        extractor: LlmExtractor | None,
        *,
        policy: ReviewPolicy | None = None,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self.scanner = scanner
        self.extractor = extractor
        self.policy = policy or ReviewPolicy()
        self.clock = clock

    def process(self, data: bytes, *, correlation_id: str | None = None) -> ProcessingResult:
        cid = correlation_id or new_correlation_id()
        state: dict[str, Any] = {
            "pipeline_version": __version__,
            "correlation_id": cid,
            "source_sha256": sha256_hex(data or b""),
            "created_at": self.clock(),
            "llm": LlmSummary(enabled=self.extractor is not None),
        }
        done: list[str] = []
        timings: dict[str, int] = {}
        record = new_record(cid)
        stage = STAGES[0]

        def timed(name: str, fn: Callable[[], Any]) -> Any:
            nonlocal stage
            stage = name
            t0 = time.monotonic()
            out = fn()
            timings[name] = int((time.monotonic() - t0) * 1000)
            done.append(name)
            return out

        try:
            prepared = timed(
                "input_validation",
                lambda: self.scanner.prepare(data, correlation_id=cid),
            )
            state["input_format"] = prepared.input_format.value
            state["preprocessing"] = {
                "mode": prepared.mode.value,
                "transformations": list(prepared.transformations),
                "warnings": list(prepared.warnings),
                "deskew_reason": prepared.deskew.reason if prepared.deskew else None,
                "deskew_angle": (
                    prepared.deskew.angle if prepared.deskew and prepared.deskew.applied else None
                ),
            }
            done.append("preprocessing")  # preparation covers validation + preprocessing

            result = timed(
                "ocr",
                lambda: self.scanner.run_ocr(prepared.payload, correlation_id=cid),
            )
            content: str = getattr(result, "content", None) or ""

            report: OcrConfidenceReport = timed(
                "confidence_assessment",
                lambda: evaluate_ocr_confidence(result, self.policy.ocr_confidence_threshold),
            )
            state["ocr"] = OcrSummary(
                model_id=self.scanner.model_id,
                page_count=report.page_count,
                scored_word_count=report.scored_word_count,
                min_confidence=report.min_confidence,
                mean_confidence=report.mean_confidence,
                threshold=report.threshold,
                flag_counts=report.flag_counts(),
            )
            state["ocr_review_items"] = tuple(
                OcrReviewItemOut(
                    page=i.page,
                    flag=i.flag.value,
                    confidence=i.confidence,
                    text=i.content,
                    polygon=i.polygon,
                )
                for i in report.review_items
            )
            if not content.strip() or report.flags & {OcrFlag.NO_PAGES}:
                raise OcrError("ocr_no_text_detected")

            extraction: ClinicalDocumentExtraction | None = None
            extractor = self.extractor
            if extractor is not None:
                annotated = annotate_low_confidence(content, _uncertain_spans(report))
                outcome = timed(
                    "llm_extraction",
                    lambda: extractor.extract(annotated, correlation_id=cid),
                )
                state["llm"] = LlmSummary(
                    enabled=True,
                    model=outcome.model,
                    prompt_version=outcome.prompt_version,
                    prompt_tokens=outcome.prompt_tokens,
                    completion_tokens=outcome.completion_tokens,
                )
                parsed = outcome.extraction
                # Re-validate defensively: the object must satisfy the schema and its validators.
                extraction = timed(
                    "validation",
                    lambda: ClinicalDocumentExtraction.model_validate(parsed.model_dump()),
                )

            record = timed(
                "review_routing",
                lambda: route_for_review(
                    record,
                    ocr=report,
                    content=content,
                    extraction=extraction,
                    policy=self.policy,
                    llm_enabled=self.extractor is not None,
                ),
            )
            final = ProcessingResult(
                **state,
                processing_status=ProcessingStatus.COMPLETED,
                stages_completed=tuple(done),
                extraction=extraction,
                review=record,
                timings_ms=timings,
            )
        except PipelineError as exc:
            final = self._failed(
                state,
                record,
                done,
                timings,
                getattr(exc, "stage", None) or stage,
                exc.code,
                exc.retryable,
            )
        except Exception as exc:  # noqa: BLE001 - boundary: unexpected bugs become a coded failure, never success
            log_event(
                logger,
                logging.ERROR,
                "pipeline_internal_error",
                correlation_id=cid,
                stage=stage,
                error_type=type(exc).__name__,
            )
            final = self._failed(state, record, done, timings, stage, "internal_error", False)

        log_event(
            logger,
            logging.INFO,
            "document_processed",
            correlation_id=cid,
            processing_status=final.processing_status.value,
            review_status=final.review.status.value,
            reason_count=len(final.review.reasons),
            error_code=final.error_code,
            pipeline_version=__version__,
        )
        return final

    @staticmethod
    def _failed(
        state: dict[str, Any],
        record: ReviewRecord,
        done: list[str],
        timings: dict[str, int],
        stage: str,
        code: str,
        retryable: bool,
    ) -> ProcessingResult:
        if record.status is not ReviewStatus.PENDING:  # failure after routing: restart the record trail
            record = new_record(record.correlation_id)
        failed_record = mark_failed(record, code)
        return ProcessingResult(
            **state,
            processing_status=ProcessingStatus.FAILED,
            stages_completed=tuple(done),
            failed_stage=stage if stage != "unknown" else None,
            error_code=code,
            retryable=retryable,
            extraction=None,
            review=failed_record,
            timings_ms=timings,
        )
