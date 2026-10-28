from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MedicationItem(BaseModel):
    """Represents an individual prescribed medication item."""

    name: Optional[str] = Field(  # noqa: UP045
        default=None,
        description="The brand or generic name of the medication.",
        examples=["Amoxicillin", "Lisinopril"],
    )
    dosage: Optional[str] = Field(  # noqa: UP045
        default=None,
        description="The strength or dosage amount of the medication.",
        examples=["500mg", "10ml"],
    )
    frequency: Optional[str] = Field(  # noqa: UP045
        default=None,
        description="How often the medication should be administered.",
        examples=["Twice daily (BID)", "Once a day"],
    )
    duration: Optional[str] = Field(  # noqa: UP045
        default=None,
        description="The duration for which the medication is prescribed.",
        examples=["7 days", "30 days"],
    )
    instructions: Optional[str] = Field(  # noqa: UP045
        default=None,
        description="Special administrative instructions (e.g., take with food).",
        examples=["Take with food", "Take on an empty stomach"],
    )


class MedicalPrescription(BaseModel):
    """Structured representation of a parsed medical prescription or clinical note."""

    patient_name: Optional[str] = Field(  # noqa: UP045
        default=None,
        description="Full name of the patient.",
        examples=["John Doe", "Jane Smith"],
    )
    age_or_dob: Optional[str] = Field(  # noqa: UP045
        default=None,
        description="Patient age or date of birth.",
        examples=["45 years old", "1980-05-12"],
    )
    prescription_date: Optional[str] = Field(  # noqa: UP045
        default=None,
        description="Date when the prescription was issued.",
        examples=["2026-06-01", "June 1, 2026"],
    )
    doctor_name: Optional[str] = Field(  # noqa: UP045
        default=None,
        description="Name of the prescribing physician.",
        examples=["Dr. Robert House, M.D."],
    )
    clinic_name: Optional[str] = Field(  # noqa: UP045
        default=None,
        description="Name of the clinic, hospital, or medical practice.",
        examples=["City General Hospital", "Metro Health Clinic"],
    )
    medications: Optional[list[MedicationItem]] = Field(  # noqa: UP045
        default=None,
        description="List of medications prescribed in the document.",
    )
    clinical_notes: Optional[str] = Field(  # noqa: UP045
        default=None,
        description="General clinical notes, diagnosis, or observations written by the clinician.",
        examples=["Patient presents with acute pharyngitis. Rest recommended."],
    )
