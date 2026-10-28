"""Accuro EMR integration status - NO adapter is implemented yet, deliberately.

Public information reviewed (October 2026): an endpoint index on a QHR developer host
(apidocs.dev.qhrtech.com/docs/resources.html) lists provider-level patient-document listing,
patient search and OAuth2 references. It does not document onboarding, webhooks/events, rate
limits, a binary document-download contract, or write-back semantics, and it is on a *dev*
host. None of it has been verified for this organisation's licence or environment.

Until the vendor confirms each capability in writing and it is exercised against a sandbox
with synthetic patients, every capability below stays UNVERIFIED, and the pipeline's
``DisabledResultSink`` refuses all write-back.

Recommended first phase: read-only retrieval (or a controlled manual export), clinician review
in this application, and manual entry of approved data into Accuro by the clinician.
"""

from __future__ import annotations

from azure_scanner.integrations.emr import CapabilityStatus, EmrCapability

SYSTEM_NAME = "accuro"

ACCURO_CAPABILITIES: dict[EmrCapability, CapabilityStatus] = {
    capability: CapabilityStatus.UNVERIFIED for capability in EmrCapability
}

VENDOR_VERIFICATION_REQUIRED: tuple[str, ...] = (
    "API/partner programme enrolment, licensing, and sandbox access with synthetic patients",
    "Supported OAuth2 grant types, scopes, token lifetime, office/tenant scoping, IP/mTLS requirements",
    "How new or updated documents are discovered (polling cursor, webhook, change feed) and guarantees",
    "Document content download contract (binary endpoint, media types, size limits, pagination)",
    "Minimum metadata available per document (patient ID, document ID, version/etag, upload time)",
    "Behaviour of document and patient IDs on patient merge / relink / document move or deletion",
    "Rate limits, fair-use policy and operational SLAs",
    "Whether write-back is supported (review status, notes, structured medications) and approvals needed",
    "Vendor and contractual position on sending document content to Azure AI services",
)
