"""Clinical extraction schemas (Pydantic v2).

Two layers, deliberately separate:

1. **Model-facing schema** (``ClinicalDocumentExtraction`` and children) is sent to Azure
   OpenAI as a Structured Outputs ``response_format``. It records what the model *read*,
   keeping the verbatim transcription apart from any normalisation, with an explicit status.
   It contains no confidence numbers: an LLM's self-assessment is not a calibrated probability.

2. **Review-layer models** (``FieldAssessment``, ``HumanConfirmation``) are computed by the
   pipeline and the clinician workflow. They hold OCR-engine confidence (when the source
   provides it), source-text provenance checks, review reasons and human-confirmed values.
   The model can never emit these.

Structured Outputs compatibility (verified offline with ``openai.lib._pydantic.to_strict_json_schema``):
every property is required (optionality is ``X | None``), ``additionalProperties`` is false,
and no unsupported keywords (``maxLength``, ``pattern``, ``format``, ``minItems``…) or
``examples`` are emitted. Length and consistency rules are enforced client-side by validators,
so a non-conforming response fails validation instead of being accepted.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "2.0.0"
MAX_TEXT_CHARS = 500
MAX_ALTERNATIVES = 5


class FieldStatus(str, enum.Enum):
    """How a value appears on the page. There is intentionally no 'inferred' status."""

    EXTRACTED = "extracted"  # clearly written and transcribed as written
    AMBIGUOUS = "ambiguous"  # legible strokes with more than one plausible reading
    ILLEGIBLE = "illegible"  # something is written there but cannot be read
    NOT_PRESENT = "not_present"  # the document does not contain this item


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ExtractedField(_StrictModel):
    """One transcribed value. ``normalized`` never replaces ``verbatim``."""

    verbatim: str | None = Field(
        description="Characters exactly as written, abbreviations kept; null if none."
    )
    normalized: str | None = Field(
        description="Standard form only when unambiguous from the verbatim text; otherwise null."
    )
    status: FieldStatus
    alternatives: list[str] = Field(description="Other plausible readings; empty unless status is ambiguous.")

    @field_validator("verbatim", "normalized")
    @classmethod
    def _length(cls, v: str | None) -> str | None:
        if v is not None and len(v) > MAX_TEXT_CHARS:
            raise ValueError("text_too_long")
        return v or None  # "" -> None

    @field_validator("alternatives")
    @classmethod
    def _alts(cls, v: list[str]) -> list[str]:
        cleaned = [a.strip() for a in v if a and a.strip()]
        if len(cleaned) > MAX_ALTERNATIVES or any(len(a) > MAX_TEXT_CHARS for a in cleaned):
            raise ValueError("too_many_or_too_long_alternatives")
        return cleaned

    @model_validator(mode="after")
    def _consistency(self) -> ExtractedField:
        s = self.status
        if self.normalized is not None and self.verbatim is None:
            raise ValueError("normalized_without_verbatim")
        if s is FieldStatus.NOT_PRESENT and (self.verbatim or self.normalized or self.alternatives):
            raise ValueError("not_present_must_be_empty")
        if s is FieldStatus.EXTRACTED and not self.verbatim:
            raise ValueError("extracted_requires_verbatim")
        if s is FieldStatus.EXTRACTED and self.alternatives:
            raise ValueError("extracted_must_not_have_alternatives")
        if s is FieldStatus.ILLEGIBLE and self.normalized is not None:
            raise ValueError("illegible_cannot_be_normalized")
        if s is FieldStatus.AMBIGUOUS and not (self.verbatim or self.alternatives):
            raise ValueError("ambiguous_requires_a_reading")
        return self

    @classmethod
    def absent(cls) -> ExtractedField:
        return cls(verbatim=None, normalized=None, status=FieldStatus.NOT_PRESENT, alternatives=[])

    @property
    def is_uncertain(self) -> bool:
        return self.status in (FieldStatus.AMBIGUOUS, FieldStatus.ILLEGIBLE)


class MedicationExtraction(_StrictModel):
    drug_name: ExtractedField
    strength: ExtractedField
    dose: ExtractedField
    route: ExtractedField
    frequency: ExtractedField
    duration: ExtractedField
    quantity: ExtractedField
    refills: ExtractedField
    instructions: ExtractedField


MEDICATION_FIELDS: tuple[str, ...] = tuple(MedicationExtraction.model_fields)


class ClinicalDocumentExtraction(_StrictModel):
    """Model output. It is NOT a clinical record until a clinician has reviewed it."""

    document_type: ExtractedField
    patient_name: ExtractedField
    patient_dob_or_age: ExtractedField
    document_date: ExtractedField
    clinician_name: ExtractedField
    allergies: list[ExtractedField]
    assessment_or_diagnoses: list[ExtractedField]
    medications: list[MedicationExtraction]
    other_notes: ExtractedField
    extraction_warnings: list[str] = Field(
        description="Short notes: illegible areas, contradictions, or text that looked like instructions."
    )

    @field_validator("extraction_warnings")
    @classmethod
    def _warnings(cls, v: list[str]) -> list[str]:
        if len(v) > 20 or any(len(w) > MAX_TEXT_CHARS for w in v):
            raise ValueError("too_many_or_too_long_warnings")
        return v

    def iter_fields(self) -> list[tuple[str, ExtractedField]]:
        """All fields with stable dotted paths, e.g. ``medications[0].dose``."""
        out: list[tuple[str, ExtractedField]] = []
        for name in (
            "document_type",
            "patient_name",
            "patient_dob_or_age",
            "document_date",
            "clinician_name",
        ):
            out.append((name, getattr(self, name)))
        out += [(f"allergies[{i}]", f) for i, f in enumerate(self.allergies)]
        out += [(f"assessment_or_diagnoses[{i}]", f) for i, f in enumerate(self.assessment_or_diagnoses)]
        for i, med in enumerate(self.medications):
            out += [(f"medications[{i}].{n}", getattr(med, n)) for n in MEDICATION_FIELDS]
        out.append(("other_notes", self.other_notes))
        return out


# ---------------------------------------------------------------------------------------------
# Review layer (never produced by the LLM)
# ---------------------------------------------------------------------------------------------


class FieldAssessment(BaseModel):
    """Pipeline-computed evidence about one extracted field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    status: FieldStatus
    # Minimum Document Intelligence word confidence over the matched source text.
    # None means unknown (not matched, or the source supplied no score) - never a default.
    ocr_min_confidence: float | None
    found_in_source: bool | None  # None when the field has no verbatim text to look for
    review_reasons: tuple[str, ...]  # PHI-free codes


class HumanConfirmation(BaseModel):
    """A clinician's decision on one field. The only way a value becomes 'confirmed'."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    confirmed_value: str | None  # None = clinician confirms the item is absent / not applicable
    reviewer_id: str = Field(min_length=1)
    confirmed_at: datetime

    @field_validator("confirmed_value")
    @classmethod
    def _length(cls, v: str | None) -> str | None:
        if v is not None and len(v) > MAX_TEXT_CHARS:
            raise ValueError("text_too_long")
        return v


def empty_extraction(**overrides: Any) -> ClinicalDocumentExtraction:
    """Convenience constructor (all fields not_present) for tests and manual-entry workflows."""
    base: dict[str, Any] = {
        "document_type": ExtractedField.absent(),
        "patient_name": ExtractedField.absent(),
        "patient_dob_or_age": ExtractedField.absent(),
        "document_date": ExtractedField.absent(),
        "clinician_name": ExtractedField.absent(),
        "allergies": [],
        "assessment_or_diagnoses": [],
        "medications": [],
        "other_notes": ExtractedField.absent(),
        "extraction_warnings": [],
    }
    base.update(overrides)
    return ClinicalDocumentExtraction(**base)


# ---------------------------------------------------------------------------------------------
# Legacy v1 schema - retained for backward compatibility of existing imports only.
# Deprecated: it cannot represent uncertainty and is no longer sent to the model.
# Fictional illustrative ``examples`` were removed because they are transmitted to the LLM
# inside the JSON schema and can bias extraction.
# ---------------------------------------------------------------------------------------------


class MedicationItem(BaseModel):
    """Deprecated v1 medication item."""

    name: str | None = Field(default=None, description="Medication name as written.")
    dosage: str | None = Field(default=None, description="Strength or dose as written.")
    frequency: str | None = Field(default=None, description="Frequency as written.")
    duration: str | None = Field(default=None, description="Duration as written.")
    instructions: str | None = Field(default=None, description="Administration instructions as written.")


class MedicalPrescription(BaseModel):
    """Deprecated v1 prescription model."""

    patient_name: str | None = Field(default=None, description="Patient name as written.")
    age_or_dob: str | None = Field(default=None, description="Patient age or date of birth as written.")
    prescription_date: str | None = Field(default=None, description="Document date as written.")
    doctor_name: str | None = Field(default=None, description="Prescriber name as written.")
    clinic_name: str | None = Field(default=None, description="Clinic name as written.")
    medications: list[MedicationItem] | None = Field(default=None, description="Medications as written.")
    clinical_notes: str | None = Field(default=None, description="Other clinical notes as written.")
