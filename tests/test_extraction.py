from __future__ import annotations

import json
from typing import Any

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from azure_scanner.models import (
    ClinicalDocumentExtraction,
    ExtractedField,
    FieldStatus,
    MedicalPrescription,
    MedicationExtraction,
    MedicationItem,
    empty_extraction,
)


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


# ---------------------------------------------------------------------------------------------
# v2 schema: uncertainty preservation, validation, serialization, Structured Outputs compatibility
# ---------------------------------------------------------------------------------------------

def _f(verbatim=None, status="not_present", normalized=None, alternatives=None):  # type: ignore[no-untyped-def]
    return {"verbatim": verbatim, "normalized": normalized, "status": status, "alternatives": alternatives or []}


@pytest.mark.parametrize(
    "data",
    [
        _f("0.5 mg", "extracted"),
        _f("0.5 mg", "extracted", normalized="0.5 mg"),
        _f("5 or 0.5 mg", "ambiguous", alternatives=["5 mg", "0.5 mg"]),
        _f(None, "ambiguous", alternatives=["bid", "tid"]),
        _f("q?h", "illegible"),
        _f(None, "illegible"),
        _f(None, "not_present"),
    ],
)
def test_valid_field_states(data: dict) -> None:  # type: ignore[type-arg]
    assert ExtractedField.model_validate(data).status.value == data["status"]


@pytest.mark.parametrize(
    "data, error",
    [
        (_f("x", "not_present"), "not_present_must_be_empty"),
        (_f(None, "not_present", alternatives=["x"]), "not_present_must_be_empty"),
        (_f(None, "extracted"), "extracted_requires_verbatim"),
        (_f("  ", "extracted"), "extracted_requires_verbatim"),
        (_f("bid", "extracted", alternatives=["tid"]), "extracted_must_not_have_alternatives"),
        (_f("q?h", "illegible", normalized="q4h"), "illegible_cannot_be_normalized"),
        (_f(None, "ambiguous"), "ambiguous_requires_a_reading"),
        (_f(None, "extracted", normalized="500 mg"), "normalized_without_verbatim"),
        (_f("x" * 501, "extracted"), "text_too_long"),
        (_f("a", "ambiguous", alternatives=[str(i) for i in range(6)]), "too_many_or_too_long_alternatives"),
        ({**_f("x", "extracted"), "status": "inferred"}, "status"),  # no inferred status exists
        ({**_f("x", "extracted"), "confidence": 0.99}, "confidence"),  # LLM cannot assert confidence
        ({"verbatim": "x", "status": "extracted"}, "normalized"),  # missing required keys
        (_f(123, "extracted"), "verbatim"),  # malformed type
    ],
)
def test_invalid_field_states_are_rejected(data: dict, error: str) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValidationError) as exc:
        ExtractedField.model_validate(data)
    assert error in str(exc.value)


def test_null_vs_missing_is_explicit() -> None:
    field = ExtractedField.model_validate(_f(None, "not_present"))
    assert field.verbatim is None and field.status is FieldStatus.NOT_PRESENT
    assert not field.is_uncertain
    assert ExtractedField.model_validate(_f(None, "illegible")).is_uncertain


def test_serialization_round_trip_is_lossless() -> None:
    med = MedicationExtraction(
        drug_name=ExtractedField.model_validate(_f("Drug-X", "extracted")),
        strength=ExtractedField.model_validate(_f("0.5mg", "ambiguous", alternatives=["5mg"])),
        dose=ExtractedField.absent(),
        route=ExtractedField.model_validate(_f("PO", "extracted", normalized="oral")),
        frequency=ExtractedField.model_validate(_f(None, "illegible")),
        duration=ExtractedField.absent(),
        quantity=ExtractedField.absent(),
        refills=ExtractedField.absent(),
        instructions=ExtractedField.absent(),
    )
    doc = empty_extraction(medications=[med], extraction_warnings=["frequency illegible"])
    again = ClinicalDocumentExtraction.model_validate_json(doc.model_dump_json())
    assert again == doc
    assert json.loads(again.model_dump_json())["medications"][0]["strength"]["alternatives"] == ["5mg"]


def test_extra_top_level_keys_rejected() -> None:
    data = json.loads(empty_extraction().model_dump_json())
    data["diagnosis_code"] = "J02"
    with pytest.raises(ValidationError):
        ClinicalDocumentExtraction.model_validate(data)


def test_iter_fields_paths() -> None:
    doc = empty_extraction(medications=[MedicationExtraction(**{n: ExtractedField.absent() for n in MedicationExtraction.model_fields})])
    paths = [p for p, _ in doc.iter_fields()]
    assert "medications[0].dose" in paths and "patient_name" in paths and len(paths) == 6 + 9


def _walk(node, depth=0):  # type: ignore[no-untyped-def]
    """Yield (object_depth, node) for every JSON-schema object node."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield depth, node
            depth += 1
        for v in node.values():
            yield from _walk(v, depth)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v, depth)


UNSUPPORTED_KEYWORDS = (
    "minLength", "maxLength", "pattern", "format", "minimum", "maximum", "multipleOf",
    "patternProperties", "unevaluatedProperties", "propertyNames", "minProperties", "maxProperties",
    "unevaluatedItems", "contains", "minContains", "maxContains", "minItems", "maxItems", "uniqueItems",
    "examples", "default",
)


def test_schema_is_structured_outputs_compatible_offline() -> None:
    """Checks the documented Azure OpenAI Structured Outputs subset. Live acceptance by the actual
    deployment is NOT verified by this test (see README: 'Azure compatibility status')."""
    schema = to_strict_json_schema(ClinicalDocumentExtraction)
    objects = list(_walk(schema))
    assert schema["type"] == "object" and "anyOf" not in schema  # root may not be anyOf
    for _, obj in objects:
        assert obj.get("additionalProperties") is False
        assert set(obj.get("required", [])) == set(obj.get("properties", {}))  # all fields required
    dumped = json.dumps(schema)
    for kw in UNSUPPORTED_KEYWORDS:
        assert f'"{kw}"' not in dumped, kw
    assert sum(len(o.get("properties", {})) for _, o in objects) <= 100
    assert max(d for d, _ in objects) + 1 <= 5  # nesting levels


def test_legacy_schema_has_no_fictional_examples() -> None:
    dumped = json.dumps(MedicalPrescription.model_json_schema())
    assert '"examples"' not in dumped
    for fictional in ("John Doe", "House", "Amoxicillin", "Lisinopril", "City General"):
        assert fictional not in dumped
