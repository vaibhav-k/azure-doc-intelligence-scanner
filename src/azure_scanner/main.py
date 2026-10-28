"""Command-line entry point.

    azure-scanner process INPUT [INPUT ...] --out-dir DIR [--no-llm] [--env-file .env]
    python -m azure_scanner.main process ...

* stdout receives one PHI-free JSON summary line per input (index, correlation ID, status,
  reason codes). Clinical content is never printed.
* Full results (which contain PHI) are written to ``DIR/<correlation_id>.json`` with owner-only
  permissions. Input file names are not reused (they often contain patient names); the mapping
  from input path to correlation ID is written to ``DIR/manifest.jsonl``, also owner-only.
* Identical inputs (same SHA-256) in one run are processed once.
* Exit codes: 0 = every input completed (still requires clinician review),
  1 = at least one input failed, 2 = usage or configuration error.
* Retention: this tool does not delete outputs. Apply your organisation's retention policy
  to ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from azure_scanner.config import Settings, load_settings
from azure_scanner.errors import PipelineError
from azure_scanner.llm_parser import LlmExtractor
from azure_scanner.logging_utils import configure_logging, log_event, new_correlation_id
from azure_scanner.pipeline import DocumentPipeline, ProcessingStatus, sha256_hex
from azure_scanner.review import ReviewPolicy
from azure_scanner.service import DocumentScannerService, build_di_client

logger = logging.getLogger("azure_scanner.cli")


def _write_private(path: Path, text: str) -> None:
    """Create a new file readable only by the owner. Never overwrites."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)


def _append_private(path: Path, line: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _emit(summary: dict[str, object]) -> None:
    """One PHI-free JSON line per input on stdout."""
    sys.stdout.write(json.dumps(summary) + "\n")


def build_pipeline(settings: Settings, *, use_llm: bool) -> DocumentPipeline:
    scanner = DocumentScannerService(
        build_di_client(settings),
        settings.azure_document_intelligence_model,
        preprocess_mode=settings.preprocess_mode,
        timeout_seconds=settings.ocr_timeout_seconds,
        max_input_bytes=settings.max_input_bytes,
    )
    extractor = LlmExtractor.from_settings(settings) if use_llm else None
    return DocumentPipeline(scanner, extractor, policy=ReviewPolicy(settings.ocr_confidence_threshold))


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="azure-scanner", description=__doc__.splitlines()[0] if __doc__ else None)
    sub = p.add_subparsers(dest="command", required=True)
    proc = sub.add_parser("process", help="process one or more documents")
    proc.add_argument("inputs", nargs="+", type=Path)
    proc.add_argument("--out-dir", type=Path, required=True, help="directory for full results (contain PHI)")
    proc.add_argument("--no-llm", action="store_true", help="OCR + confidence only, even if ENABLE_LLM=true")
    proc.add_argument("--env-file", default=".env", help="optional .env file (default: .env if present)")
    return p


def main(argv: list[str] | None = None, *, pipeline: DocumentPipeline | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:  # argparse exits on --help / usage errors
        return 0 if exc.code in (0, None) else 2

    if pipeline is None:
        env_file = args.env_file if args.env_file and Path(args.env_file).is_file() else None
        try:
            settings = load_settings(env_file)
        except PipelineError as exc:
            sys.stderr.write(f"configuration error: {exc.detail}\n")  # variable names only
            return 2
        configure_logging(settings.log_level)
        pipeline = build_pipeline(settings, use_llm=settings.enable_llm and not args.no_llm)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = out_dir / "manifest.jsonl"
    seen: dict[str, str] = {}
    any_failed = False

    for index, path in enumerate(args.inputs):
        cid = new_correlation_id()
        try:
            data = path.read_bytes()
        except OSError:
            any_failed = True
            log_event(logger, logging.ERROR, "input_unreadable", correlation_id=cid, error_code="input_unreadable")
            _emit({"index": index, "correlation_id": cid, "processing_status": "failed",
                              "error_code": "input_unreadable"})
            continue
        digest = sha256_hex(data)
        if digest in seen:
            _emit({"index": index, "duplicate_of": seen[digest], "processing_status": "skipped_duplicate"})
            continue
        seen[digest] = cid

        result = pipeline.process(data, correlation_id=cid)
        _write_private(out_dir / f"{cid}.json", result.model_dump_json(indent=2))
        _append_private(manifest, json.dumps({"index": index, "input_path": str(path), "correlation_id": cid,
                                              "source_sha256": digest}))
        any_failed |= result.processing_status is ProcessingStatus.FAILED
        _emit({"index": index, **result.safe_summary()})

    return 1 if any_failed else 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
