from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from azure_scanner.confidence import evaluate_ocr_confidence
from azure_scanner.models import HumanConfirmation
from azure_scanner.review import (
    ReviewPolicy,
    ReviewPriority,
    ReviewRecord,
    ReviewStatus,
    ReviewTransitionError,
    approve,
    assess_field,
    mark_failed,
    new_record,
    reject,
    required_confirmation_paths,
    route_for_review,
)
from tests.conftest import PHI_SENTINELS, ocr_result_from_text
from tests.factories import CLEAN_TEXT, clean_med, doc, fld, med

POLICY = ReviewPolicy()
T0 = datetime(2026, 10, 28, tzinfo=timezone.utc)


def _route(extraction, text=CLEAN_TEXT, confidences=None, llm_enabled=True) -> ReviewRecord:  # type: ignore[no-untyped-def]
    ocr = evaluate_ocr_confidence(ocr_result_from_text(text, confidences))
    return route_for_review(
        new_record("c1"),
        ocr=ocr,
        content=text,
        extraction=extraction,
        policy=POLICY,
        llm_enabled=llm_enabled,
    )


def _confirm(record: ReviewRecord, reviewer: str = "dr-reviewer-01") -> list[HumanConfirmation]:
    return [
        HumanConfirmation(path=p, confirmed_value="ok", reviewer_id=reviewer, confirmed_at=T0)
        for p in required_confirmation_paths(record, POLICY)
    ]


# ---------------------------------------------------------------- routing never approves


def test_schema_valid_clean_extraction_still_needs_review() -> None:
    record = _route(doc(medications=[clean_med()]))
    assert record.status is ReviewStatus.NEEDS_REVIEW
    assert record.priority is ReviewPriority.ROUTINE
    assert ReviewStatus.APPROVED not in {e.to_status for e in record.events}


@pytest.mark.parametrize(
    "field_name, value, reason",
    [
        (
            "drug_name",
            fld("Drug-X or Drug-Y", "ambiguous", alternatives=["Drug-X", "Drug-Y"]),
            "ambiguous",
        ),
        ("strength", fld(None, "illegible"), "illegible"),
        (
            "dose",
            fld("1 or 7 tab", "ambiguous", alternatives=["1 tab", "7 tab"]),
            "ambiguous",
        ),
        ("route", fld(None, "illegible"), "illegible"),
        ("frequency", fld("q?h", "illegible"), "illegible"),
        (
            "duration",
            fld("7 or 14 d", "ambiguous", alternatives=["7 d", "14 d"]),
            "ambiguous",
        ),
    ],
)
def test_uncertain_medication_elements_route_to_priority_review(field_name: str, value, reason: str) -> None:  # type: ignore[no-untyped-def]
    m = clean_med().model_copy(update={field_name: value})
    record = _route(doc(medications=[m]))
    assert record.status is ReviewStatus.NEEDS_REVIEW and record.priority is ReviewPriority.PRIORITY
    flagged = {a.path: a.review_reasons for a in record.field_assessments}
    assert reason in flagged[f"medications[0].{field_name}"]


@pytest.mark.parametrize("missing", ["drug_name", "strength", "frequency"])
def test_missing_core_medication_elements_are_review_reasons(missing: str) -> None:
    m = clean_med().model_copy(update={missing: fld()})
    assert f"medication_{missing}_missing" in _route(doc(medications=[m])).reasons


def test_low_ocr_confidence_on_a_dose_is_flagged_on_that_field() -> None:
    record = _route(doc(medications=[clean_med()]), confidences={"2": 0.31})
    strength = next(a for a in record.field_assessments if a.path == "medications[0].strength")
    assert strength.ocr_min_confidence == 0.31
    assert "ocr_low_confidence" in strength.review_reasons
    assert record.priority is ReviewPriority.PRIORITY


def test_unknown_ocr_confidence_is_not_treated_as_high() -> None:
    record = _route(doc(medications=[clean_med()]), confidences={"BID": None})
    freq = next(a for a in record.field_assessments if a.path == "medications[0].frequency")
    assert freq.ocr_min_confidence is None
    assert "ocr_confidence_unknown" in freq.review_reasons


def test_value_absent_from_ocr_source_is_flagged_as_possible_hallucination() -> None:
    m = clean_med().model_copy(update={"strength": fld("20 mg")})  # OCR says "2 mg"
    record = _route(doc(medications=[m]))
    a = next(a for a in record.field_assessments if a.path == "medications[0].strength")
    assert a.found_in_source is False and "not_found_in_source" in a.review_reasons
    assert record.priority is ReviewPriority.PRIORITY


def test_normalization_that_changes_numbers_is_flagged() -> None:
    m = clean_med().model_copy(update={"strength": fld("2 mg", normalized="20 mg")})
    assert "normalization_changed_numbers" in _route(doc(medications=[m])).reasons


@pytest.mark.parametrize("text", [".5 mg", "5.0 mg"])
def test_error_prone_decimal_notation_flagged(text: str) -> None:
    m = clean_med().model_copy(update={"strength": fld(text)})
    record = _route(doc(medications=[m]), text=f"Rx Drug-X {text} PO BID 7 days")
    assert "error_prone_decimal" in record.reasons


def test_contradictory_entries_for_same_drug_flagged() -> None:
    record = _route(
        doc(
            medications=[
                clean_med(),
                clean_med().model_copy(update={"strength": fld("20 mg")}),
            ]
        ),
        text=CLEAN_TEXT + " Drug-X 20 mg",
    )
    assert "contradictory_medication_entries" in record.reasons


def test_dose_without_drug_name_flagged() -> None:
    record = _route(doc(medications=[med(strength="2 mg", frequency="BID")]))
    assert "dose_without_drug_name" in record.reasons


def test_uncertain_allergy_and_identity_flagged() -> None:
    record = _route(doc(allergies=[fld(None, "illegible")], patient_name=fld(None, "illegible")))
    assert {"allergy_uncertain", "identity_uncertain"} <= set(record.reasons)


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"confidences": {"BID": None}}, "ocr_confidence_unknown"),
        (
            {"extraction": doc(medications=[clean_med().model_copy(update={"strength": fld()})])},
            "medication_strength_missing",
        ),
        (
            {
                "extraction": doc(
                    medications=[clean_med()],
                    extraction_warnings=["text asked to ignore rules"],
                )
            },
            "model_reported_warnings",
        ),
    ],
)
def test_every_reason_escalates_to_priority_by_default(kwargs: dict, reason: str) -> None:  # type: ignore[type-arg]
    extraction = kwargs.pop("extraction", doc(medications=[clean_med()]))
    record = _route(extraction, **kwargs)
    assert reason in record.reasons and record.priority is ReviewPriority.PRIORITY


def test_routine_reasons_are_an_explicit_opt_in() -> None:
    m = clean_med().model_copy(update={"strength": fld()})
    ocr = evaluate_ocr_confidence(ocr_result_from_text(CLEAN_TEXT))
    policy = ReviewPolicy(routine_reasons=frozenset({"medication_strength_missing"}))
    record = route_for_review(
        new_record("c"),
        ocr=ocr,
        content=CLEAN_TEXT,
        extraction=doc(medications=[m]),
        policy=policy,
        llm_enabled=True,
    )
    assert record.priority is ReviewPriority.ROUTINE and record.status is ReviewStatus.NEEDS_REVIEW


def test_ocr_document_flags_propagate() -> None:
    record = _route(doc(), confidences={"Rx": 0.2})
    assert "ocr_low_confidence" in record.reasons


def test_llm_disabled_requires_manual_transcription_and_cannot_be_approved() -> None:
    record = _route(None, llm_enabled=False)
    assert record.status is ReviewStatus.NEEDS_REVIEW
    assert "llm_disabled_manual_transcription_required" in record.reasons
    with pytest.raises(ReviewTransitionError, match="manual_transcription"):
        approve(record, reviewer_id="dr-1", confirmations=[])


def test_assess_field_not_present_has_unknown_provenance() -> None:
    ocr = evaluate_ocr_confidence(ocr_result_from_text(CLEAN_TEXT))
    a = assess_field("route", fld(), ocr, CLEAN_TEXT, POLICY)
    assert a.found_in_source is None and a.ocr_min_confidence is None and a.review_reasons == ()


# ---------------------------------------------------------------- state machine


def test_approval_requires_confirmation_of_every_consequential_field() -> None:
    record = _route(doc(medications=[clean_med()]))
    required = required_confirmation_paths(record, POLICY)
    assert {
        "medications[0].drug_name",
        "medications[0].strength",
        "medications[0].frequency",
    } <= required
    with pytest.raises(ReviewTransitionError, match="unconfirmed_fields"):
        approve(record, reviewer_id="dr-reviewer-01", confirmations=_confirm(record)[:-1])
    approved = approve(
        record,
        reviewer_id="dr-reviewer-01",
        confirmations=_confirm(record),
        acknowledged_reasons=record.reasons,
        at=T0,
    )
    assert approved.status is ReviewStatus.APPROVED
    assert approved.events[-1].actor_type == "clinician" and approved.events[-1].actor_id == "dr-reviewer-01"
    assert record.status is ReviewStatus.NEEDS_REVIEW  # records are immutable


def test_document_level_reasons_must_be_acknowledged() -> None:
    # Low-confidence OCR text that maps to no extracted field (e.g. an omitted medication line).
    record = _route(
        doc(medications=[clean_med()]),
        text=CLEAN_TEXT + " Drug-Z 40 mg",
        confidences={"Drug-Z": 0.3},
    )
    assert "ocr_low_confidence" in record.reasons
    with pytest.raises(ReviewTransitionError, match="unacknowledged_reasons"):
        approve(record, reviewer_id="dr-reviewer-01", confirmations=_confirm(record))
    ok = approve(
        record,
        reviewer_id="dr-reviewer-01",
        confirmations=_confirm(record),
        acknowledged_reasons=record.reasons,
    )
    assert ok.status is ReviewStatus.APPROVED
    assert set(ok.events[-1].reason_codes) == set(record.reasons)


def test_confirmations_must_belong_to_approving_reviewer() -> None:
    record = _route(doc(medications=[clean_med()]))
    with pytest.raises(ReviewTransitionError, match="reviewer_mismatch"):
        approve(record, reviewer_id="dr-a", confirmations=_confirm(record, "dr-b"))


def test_approval_requires_reviewer() -> None:
    with pytest.raises(ReviewTransitionError, match="reviewer_required"):
        approve(_route(doc()), reviewer_id="", confirmations=[])


@pytest.mark.parametrize("start", ["pending", "failed", "approved", "rejected"])
def test_only_needs_review_can_be_approved(start: str) -> None:
    rec = new_record("c")
    if start == "failed":
        rec = mark_failed(rec, "ocr_timeout")
    elif start in ("approved", "rejected"):
        rec = _route(doc())
        rec = (
            approve(rec, reviewer_id="d", confirmations=[])
            if start == "approved"
            else reject(rec, reviewer_id="d", reason_code="illegible_source")
        )
    with pytest.raises(ReviewTransitionError):
        approve(rec, reviewer_id="d", confirmations=[])


def test_failed_cannot_transition_anywhere() -> None:
    failed = mark_failed(new_record("c"), "ocr_timeout")
    assert failed.status is ReviewStatus.FAILED
    with pytest.raises(ReviewTransitionError):
        reject(failed, reviewer_id="d", reason_code="x")


def test_reject_requires_code_not_free_text() -> None:
    with pytest.raises(ReviewTransitionError, match="reason_code_must_be_a_code"):
        reject(
            _route(doc()),
            reviewer_id="d",
            reason_code=f"patient {PHI_SENTINELS[0]} wrong chart",
        )


def test_status_changes_are_recorded_and_logged_without_phi(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    m = clean_med().model_copy(update={"drug_name": fld(PHI_SENTINELS[2])})
    record = _route(
        doc(patient_name=fld(PHI_SENTINELS[0]), medications=[m]),
        text=f"{PHI_SENTINELS[0]} {CLEAN_TEXT} {PHI_SENTINELS[2]}",
    )
    record = reject(record, reviewer_id="dr-reviewer-01", reason_code="wrong_patient", at=T0)
    assert [(e.from_status.value, e.to_status.value) for e in record.events] == [
        ("pending", "needs_review"),
        ("needs_review", "rejected"),
    ]
    logged = " ".join(r.getMessage() + repr(getattr(r, "safe_fields", "")) for r in caplog.records)
    assert "review_status_changed" in logged
    assert all(s not in logged for s in PHI_SENTINELS) and "dr-reviewer-01" not in logged
    assert all(s not in " ".join(record.reasons) for s in PHI_SENTINELS)
