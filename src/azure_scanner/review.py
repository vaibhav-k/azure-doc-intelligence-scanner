"""Clinician-review routing and the review state machine.

Principles
----------
* There is no automatic approval. A successfully processed document always enters
  ``needs_review``; only a named clinician can move it to ``approved`` or ``rejected``.
* Routing uses checkable evidence (OCR-engine confidence, presence of the value in the OCR
  source text, field status, deterministic clinical-safety rules). LLM self-assessment is not
  used and no confidence number is invented.
* Approval requires an explicit human confirmation for every flagged field and every
  clinically consequential field (medication elements and allergies). Schema validity alone
  never suffices.
* Reason codes and review events are PHI-free; reviewer identity is stored on the record,
  not written to logs.
"""

from __future__ import annotations

import enum
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from azure_scanner.confidence import OcrConfidenceReport, OcrFlag, words_in_span
from azure_scanner.errors import PipelineError
from azure_scanner.logging_utils import log_event
from azure_scanner.models import (
    ClinicalDocumentExtraction,
    ExtractedField,
    FieldAssessment,
    FieldStatus,
    HumanConfirmation,
)

logger = logging.getLogger(__name__)


class ReviewStatus(str, enum.Enum):
    PENDING = "pending"  # created, processing not finished
    NEEDS_REVIEW = "needs_review"  # processing completed; awaiting clinician
    APPROVED = "approved"  # clinician confirmed (terminal)
    REJECTED = "rejected"  # clinician rejected (terminal)
    FAILED = "failed"  # processing failed; nothing to review (terminal)


class ReviewPriority(str, enum.Enum):
    ROUTINE = "routine"
    PRIORITY = "priority"


ALLOWED_TRANSITIONS: dict[ReviewStatus, frozenset[ReviewStatus]] = {
    ReviewStatus.PENDING: frozenset({ReviewStatus.NEEDS_REVIEW, ReviewStatus.FAILED}),
    ReviewStatus.NEEDS_REVIEW: frozenset({ReviewStatus.APPROVED, ReviewStatus.REJECTED}),
    ReviewStatus.APPROVED: frozenset(),
    ReviewStatus.REJECTED: frozenset(),
    ReviewStatus.FAILED: frozenset(),
}


class ReviewTransitionError(PipelineError):
    stage = "review"


@dataclass(frozen=True)
class ReviewPolicy:
    """Configurable routing rules. Defaults are conservative; tighten with your clinical lead."""

    ocr_confidence_threshold: float = 0.85
    # Fields whose value must be explicitly confirmed by a clinician before approval.
    mandatory_confirmation_prefixes: tuple[str, ...] = ("medications[", "allergies[")
    # Medication elements whose absence is itself a review reason.
    required_medication_fields: tuple[str, ...] = ("drug_name", "strength", "frequency")
    # Fields where error-prone decimal notation is flagged (ISMP: trailing zero / naked decimal).
    decimal_checked_fields: tuple[str, ...] = ("strength", "dose", "quantity")
    # Uncertain patient identifiers matter for patient-document association.
    identity_fields: tuple[str, ...] = ("patient_name", "patient_dob_or_age")
    # Fail-safe escalation: ANY review reason makes the document ``priority`` unless the reason is
    # listed here. Empty by default; add codes only with clinical sign-off.
    routine_reasons: frozenset[str] = field(default_factory=frozenset[str])


_TRAILING_ZERO = re.compile(r"\b\d+\.0+\b")  # "5.0 mg" can be misread as 50 mg
_NAKED_DECIMAL = re.compile(r"(?<![\d.])\.\d")  # ".5 mg" can be misread as 5 mg
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_WS = re.compile(r"\s+")


def _numbers(text: str | None) -> list[str]:
    return [n.replace(",", ".") for n in _NUMBER.findall(text or "")]


def _find_in_source(needle: str, content: str) -> list[tuple[int, int]]:
    """Locate ``needle`` in OCR content, tolerant of whitespace and case. Returns (offset, length)."""
    tokens = [re.escape(t) for t in needle.split()]
    if not tokens:
        return []
    pattern = re.compile(r"\s+".join(tokens), re.IGNORECASE)
    return [(m.start(), m.end() - m.start()) for m in pattern.finditer(content)]


def _check_status(value: ExtractedField, reasons: list[str]) -> None:
    if value.status in (FieldStatus.AMBIGUOUS, FieldStatus.ILLEGIBLE):
        reasons.append(value.status.value)


def _check_source_and_confidence(
    value: ExtractedField,
    content: str,
    ocr: OcrConfidenceReport,
    policy: ReviewPolicy,
    reasons: list[str],
) -> tuple[bool | None, float | None]:
    if not value.verbatim:
        return None, None

    hits = _find_in_source(value.verbatim, content)
    found = bool(hits)
    if not found:
        reasons.append("not_found_in_source")
        return found, None

    words = [w for off, n in hits for w in words_in_span(ocr, off, n)]
    known = [w.confidence for w in words if w.confidence is not None]

    if not words or len(known) < len(words):
        reasons.append("ocr_confidence_unknown")

    min_conf = min(known) if known else None
    if min_conf is not None and min_conf < policy.ocr_confidence_threshold:
        reasons.append("ocr_low_confidence")

    return found, min_conf


def _check_formatting_and_identity(
    path: str, value: ExtractedField, policy: ReviewPolicy, reasons: list[str]
) -> None:
    leaf = path.rsplit(".", 1)[-1]

    if value.normalized is not None and _numbers(value.normalized) != _numbers(value.verbatim):
        reasons.append("normalization_changed_numbers")

    if leaf in policy.decimal_checked_fields and value.verbatim:
        if _TRAILING_ZERO.search(value.verbatim) or _NAKED_DECIMAL.search(value.verbatim):
            reasons.append("error_prone_decimal")

    if path in policy.identity_fields and value.is_uncertain:
        reasons.append("identity_uncertain")


def assess_field(
    path: str,
    value: ExtractedField,
    ocr: OcrConfidenceReport,
    content: str,
    policy: ReviewPolicy,
) -> FieldAssessment:
    reasons: list[str] = []

    _check_status(value, reasons)
    found, min_conf = _check_source_and_confidence(value, content, ocr, policy, reasons)
    _check_formatting_and_identity(path, value, policy, reasons)

    return FieldAssessment(
        path=path,
        status=value.status,
        ocr_min_confidence=min_conf,
        found_in_source=found,
        review_reasons=tuple(dict.fromkeys(reasons)),
    )


def _medication_reasons(extraction: ClinicalDocumentExtraction, policy: ReviewPolicy) -> list[str]:
    reasons: list[str] = []
    strengths: dict[str, set[str]] = {}
    for med in extraction.medications:
        for name in policy.required_medication_fields:
            if getattr(med, name).status is FieldStatus.NOT_PRESENT:
                reasons.append(f"medication_{name}_missing")
        if med.drug_name.status is FieldStatus.NOT_PRESENT and (med.strength.verbatim or med.dose.verbatim):
            reasons.append("dose_without_drug_name")
        if med.drug_name.verbatim and med.strength.verbatim:
            key = _WS.sub(" ", med.drug_name.verbatim.lower())
            strengths.setdefault(key, set()).add(_WS.sub("", med.strength.verbatim.lower()))
    if any(len(v) > 1 for v in strengths.values()):
        reasons.append("contradictory_medication_entries")
    return reasons


class ReviewEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    from_status: ReviewStatus
    to_status: ReviewStatus
    actor_type: str  # "system" | "clinician"
    actor_id: str  # "pipeline" for system events
    at: datetime
    reason_codes: tuple[str, ...] = ()


class ReviewRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    correlation_id: str
    status: ReviewStatus
    priority: ReviewPriority | None = None
    reasons: tuple[str, ...] = ()
    field_assessments: tuple[FieldAssessment, ...] = ()
    confirmations: tuple[HumanConfirmation, ...] = ()
    events: tuple[ReviewEvent, ...] = Field(default_factory=tuple)

    def flagged_paths(self) -> list[str]:
        return [a.path for a in self.field_assessments if a.review_reasons]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _transition(
    record: ReviewRecord,
    to: ReviewStatus,
    *,
    actor_type: str,
    actor_id: str,
    event_reasons: Iterable[str] = (),
    at: datetime | None = None,
    **updates: object,
) -> ReviewRecord:
    if to not in ALLOWED_TRANSITIONS[record.status]:
        raise ReviewTransitionError(f"illegal_transition_{record.status.value}_to_{to.value}")
    event = ReviewEvent(
        from_status=record.status,
        to_status=to,
        actor_type=actor_type,
        actor_id=actor_id,
        at=at or _now(),
        reason_codes=tuple(event_reasons),
    )
    log_event(
        logger,
        logging.INFO,
        "review_status_changed",
        correlation_id=record.correlation_id,
        from_status=record.status.value,
        to_status=to.value,
        actor_type=actor_type,
        reason_count=len(event.reason_codes),
    )
    return record.model_copy(update={"status": to, "events": (*record.events, event), **updates})


def new_record(correlation_id: str) -> ReviewRecord:
    return ReviewRecord(correlation_id=correlation_id, status=ReviewStatus.PENDING)


def mark_failed(record: ReviewRecord, error_code: str) -> ReviewRecord:
    return _transition(
        record,
        ReviewStatus.FAILED,
        actor_type="system",
        actor_id="pipeline",
        event_reasons=(error_code,),
    )


def route_for_review(
    record: ReviewRecord,
    *,
    ocr: OcrConfidenceReport,
    content: str,
    extraction: ClinicalDocumentExtraction | None,
    policy: ReviewPolicy,
    llm_enabled: bool,
) -> ReviewRecord:
    """Compute evidence and move PENDING -> NEEDS_REVIEW. Never approves."""
    doc_reasons: list[str] = []
    for flag in sorted(ocr.flags, key=lambda f: f.value):
        doc_reasons.append("ocr_low_confidence" if flag is OcrFlag.LOW_CONFIDENCE else f"ocr_{flag.value}")
    assessments: tuple[FieldAssessment, ...] = ()
    if extraction is None:
        doc_reasons.append(
            "llm_disabled_manual_transcription_required" if not llm_enabled else "extraction_unavailable"
        )
    else:
        assessments = tuple(assess_field(p, f, ocr, content, policy) for p, f in extraction.iter_fields())
        if extraction.extraction_warnings:
            doc_reasons.append("model_reported_warnings")
        doc_reasons += _medication_reasons(extraction, policy)
        if extraction.allergies and any(a.status is not FieldStatus.EXTRACTED for a in extraction.allergies):
            doc_reasons.append("allergy_uncertain")

    all_reasons = list(dict.fromkeys(doc_reasons + [r for a in assessments for r in a.review_reasons]))
    priority = (
        ReviewPriority.PRIORITY
        if any(r not in policy.routine_reasons for r in all_reasons)
        else ReviewPriority.ROUTINE
    )
    reasons = tuple(sorted(set(all_reasons)))
    return _transition(
        record,
        ReviewStatus.NEEDS_REVIEW,
        actor_type="system",
        actor_id="pipeline",
        event_reasons=reasons,
        priority=priority,
        reasons=reasons,
        field_assessments=assessments,
    )


def required_confirmation_paths(record: ReviewRecord, policy: ReviewPolicy) -> set[str]:
    paths = set(record.flagged_paths())
    for a in record.field_assessments:
        if (
            a.path.startswith(policy.mandatory_confirmation_prefixes)
            and a.status is not FieldStatus.NOT_PRESENT
        ):
            paths.add(a.path)
    return paths


def approve(
    record: ReviewRecord,
    *,
    reviewer_id: str,
    confirmations: Iterable[HumanConfirmation],
    acknowledged_reasons: Iterable[str] = (),
    policy: ReviewPolicy | None = None,
    at: datetime | None = None,
) -> ReviewRecord:
    """Clinician approval.

    Requires (a) a confirmation for every flagged or clinically consequential field and (b) an
    explicit acknowledgement of every document-level reason code - e.g. low-confidence OCR text
    that the model did not map to any field could be an omitted medication line.
    """
    policy = policy or ReviewPolicy()
    if not reviewer_id:
        raise ReviewTransitionError("reviewer_required")
    if record.status is not ReviewStatus.NEEDS_REVIEW:
        raise ReviewTransitionError(f"illegal_transition_{record.status.value}_to_approved")
    if (
        "llm_disabled_manual_transcription_required" in record.reasons
        or "extraction_unavailable" in record.reasons
    ):
        raise ReviewTransitionError("nothing_to_approve_manual_transcription_required")
    confirmations = tuple(confirmations)
    if any(c.reviewer_id != reviewer_id for c in confirmations):
        raise ReviewTransitionError("confirmation_reviewer_mismatch")
    missing = required_confirmation_paths(record, policy) - {c.path for c in confirmations}
    if missing:
        raise ReviewTransitionError("unconfirmed_fields", detail=str(len(missing)))
    unacknowledged = set(record.reasons) - set(acknowledged_reasons)
    if unacknowledged:
        raise ReviewTransitionError("unacknowledged_reasons", detail=str(len(unacknowledged)))
    return _transition(
        record,
        ReviewStatus.APPROVED,
        actor_type="clinician",
        actor_id=reviewer_id,
        event_reasons=tuple(sorted(set(acknowledged_reasons))),
        at=at,
        confirmations=confirmations,
    )


def reject(
    record: ReviewRecord,
    *,
    reviewer_id: str,
    reason_code: str,
    at: datetime | None = None,
) -> ReviewRecord:
    if not reviewer_id:
        raise ReviewTransitionError("reviewer_required")
    if not re.fullmatch(r"[a-z0-9_]{1,64}", reason_code):
        raise ReviewTransitionError("reason_code_must_be_a_code")  # prevents free-text PHI in events
    return _transition(
        record,
        ReviewStatus.REJECTED,
        actor_type="clinician",
        actor_id=reviewer_id,
        event_reasons=(reason_code,),
        at=at,
    )
