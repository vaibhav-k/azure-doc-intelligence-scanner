from __future__ import annotations

from typing import Any

from azure_scanner.models import MedicalPrescription, MedicationItem


def test_medication_item_valid() -> None:
    data = {
        "name": "Amoxicillin",
        "dosage": "500mg",
        "frequency": "Twice daily",
        "duration": "7 days",
        "instructions": "Take with food",
    }
    item = MedicationItem(**data)
    assert item.name == "Amoxicillin"
    assert item.dosage == "500mg"
    assert item.frequency == "Twice daily"
    assert item.duration == "7 days"
    assert item.instructions == "Take with food"


def test_medication_item_partial() -> None:
    item = MedicationItem(name="Aspirin")
    assert item.name == "Aspirin"
    assert item.dosage is None
    assert item.frequency is None
    assert item.duration is None
    assert item.instructions is None


def test_medical_prescription_valid() -> None:
    data: dict[str, Any] = {
        "patient_name": "Jane Doe",
        "age_or_dob": "34",
        "prescription_date": "2026-06-01",
        "doctor_name": "Dr. House",
        "clinic_name": "Princeton Plainsboro",
        "medications": [
            {
                "name": "Ibuprofen",
                "dosage": "400mg",
                "frequency": "As needed",
                "duration": "5 days",
                "instructions": "For pain",
            }
        ],
        "clinical_notes": "Mild inflammation.",
    }
    prescription = MedicalPrescription(**data)
    assert prescription.patient_name == "Jane Doe"
    assert prescription.medications is not None
    assert len(prescription.medications) == 1
    assert prescription.medications[0].name == "Ibuprofen"
    assert prescription.clinical_notes == "Mild inflammation."


def test_medical_prescription_empty_and_optional() -> None:
    prescription = MedicalPrescription()
    assert prescription.patient_name is None
    assert prescription.age_or_dob is None
    assert prescription.prescription_date is None
    assert prescription.doctor_name is None
    assert prescription.clinic_name is None
    assert prescription.medications is None
    assert prescription.clinical_notes is None
