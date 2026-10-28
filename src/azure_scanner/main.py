"""Interim entry point (replaced by the pipeline-based CLI in a later commit)."""

from __future__ import annotations

import logging
import sys

from azure_scanner.confidence import evaluate_ocr_confidence
from azure_scanner.config import load_settings
from azure_scanner.errors import PipelineError
from azure_scanner.logging_utils import configure_logging, log_event, new_correlation_id
from azure_scanner.service import DocumentScannerService, build_di_client

logger = logging.getLogger("azure_scanner.cli")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        sys.stderr.write("usage: python -m azure_scanner.main <document>\n")
        return 2
    try:
        settings = load_settings()
    except PipelineError as exc:
        sys.stderr.write(f"configuration error: {exc.detail}\n")
        return 2
    configure_logging(settings.log_level)
    cid = new_correlation_id()
    service = DocumentScannerService(
        build_di_client(settings),
        settings.azure_document_intelligence_model,
        preprocess_mode=settings.preprocess_mode,
        timeout_seconds=settings.ocr_timeout_seconds,
    )
    try:
        result = service.analyze_document(args[0], correlation_id=cid)
    except FileNotFoundError:
        log_event(logger, logging.ERROR, "input_not_found", correlation_id=cid, error_code="input_not_found")
        return 1
    except PipelineError as exc:
        log_event(logger, logging.ERROR, "processing_failed", correlation_id=cid, error_code=exc.code)
        return 1
    report = evaluate_ocr_confidence(result, settings.ocr_confidence_threshold)
    log_event(logger, logging.INFO, "ocr_assessed", correlation_id=cid, flag_counts=report.flag_counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
