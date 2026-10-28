"""Builders for synthetic extraction objects used across review / pipeline tests."""

from __future__ import annotations

from typing import Any

from azure_scanner.models import (
    MEDICATION_FIELDS,
    ClinicalDocumentExtraction,
    ExtractedField,
    MedicationExtraction,
    empty_extraction,
)


def fld(
    verbatim: str | None = None,
    status: str | None = None,
    normalized: str | None = None,
    alternatives: list[str] | None = None,
) -> ExtractedField:
    status = status or ("extracted" if verbatim else "not_present")
    return ExtractedField(
        verbatim=verbatim,
        normalized=normalized,
        status=status,  # type: ignore[arg-type]
        alternatives=alternatives or [],
    )


def med(**fields: ExtractedField | str) -> MedicationExtraction:
    values: dict[str, Any] = {n: ExtractedField.absent() for n in MEDICATION_FIELDS}
    for k, v in fields.items():
        values[k] = fld(v) if isinstance(v, str) else v
    return MedicationExtraction(**values)


def doc(**kw: Any) -> ClinicalDocumentExtraction:
    return empty_extraction(**kw)


# A clean, fully legible synthetic prescription line and its matching OCR text.
CLEAN_TEXT = "Rx Drug-X 2 mg PO BID 7 days"


def clean_med() -> MedicationExtraction:
    return med(drug_name="Drug-X", strength="2 mg", route="PO", frequency="BID", duration="7 days")
