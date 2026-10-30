"""
Command-line entry point.

    azure-scanner process INPUT [INPUT ...] --out-dir DIR [--no-llm] [--env-file .env]
    python -m azure_scanner.main process ...

* INPUT may be a file, a folder (its files, not recursive) or a wildcard such as ``scans/*.jpg``;
  wildcards are expanded here because Windows shells pass them through literally.

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
import glob
import json
import logging
import os
import sys
from pathlib import Path

from azure_scanner.config import Settings, load_settings, unknown_setting_names
from azure_scanner.credentials import build_credential, needs_entra_credential
from azure_scanner.errors import PipelineError
from azure_scanner.llm_parser import LlmExtractor
from azure_scanner.logging_utils import configure_logging, log_event, new_correlation_id
from azure_scanner.pipeline import DocumentPipeline, ProcessingStatus, sha256_hex
from azure_scanner.preprocessing import DEFAULT_MAX_INPUT_BYTES
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
    """Wire the pipeline. One Entra ID credential is shared by both Azure clients (if any is keyless)."""
    credential = build_credential(settings) if needs_entra_credential(settings, use_llm=use_llm) else None
    scanner = DocumentScannerService(
        build_di_client(settings, credential),
        settings.azure_document_intelligence_model,
        preprocess_mode=settings.preprocess_mode,
        timeout_seconds=settings.ocr_timeout_seconds,
        max_input_bytes=settings.max_input_bytes,
    )
    extractor = LlmExtractor.from_settings(settings, credential=credential) if use_llm else None
    return DocumentPipeline(
        scanner,
        extractor,
        policy=ReviewPolicy(settings.ocr_confidence_threshold),
        resources=(credential,) if credential is not None else (),
    )


_GLOB_CHARS = frozenset("*?[")


def expand_inputs(paths: list[Path]) -> list[Path]:
    """Folders become their (non-hidden) files; unmatched wildcards stay literal and fail as unreadable.

    An existing path is always taken literally, so names such as ``[clean] note.jpeg`` still work.
    """
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out += sorted(p for p in path.iterdir() if p.is_file() and not p.name.startswith("."))
        elif not path.exists() and _GLOB_CHARS & set(str(path)):
            matches = sorted(Path(m) for m in glob.glob(str(path)) if Path(m).is_file())
            out += matches or [path]
        else:
            out.append(path)
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="azure-scanner", description=__doc__.splitlines()[0] if __doc__ else None
    )
    sub = p.add_subparsers(dest="command", required=True)
    proc = sub.add_parser("process", help="process one or more documents")
    proc.add_argument("inputs", nargs="+", type=Path)
    proc.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="directory for full results (contain PHI)",
    )
    proc.add_argument(
        "--no-llm",
        action="store_true",
        help="OCR + confidence only, even if ENABLE_LLM=true",
    )
    proc.add_argument(
        "--env-file",
        default=None,
        help="settings file (default: .env in the current folder, if present). An explicit file must exist.",
    )
    return p


def _fail_item(index: int, cid: str, code: str) -> None:
    log_event(logger, logging.ERROR, code, correlation_id=cid, error_code=code)
    _emit(
        {
            "index": index,
            "correlation_id": cid,
            "processing_status": "failed",
            "error_code": code,
        }
    )


def _process_item(
    index: int,
    path: Path,
    pipeline: DocumentPipeline,
    out_dir: Path,
    manifest: Path,
    seen: dict[str, str],
) -> tuple[bool, str | None]:
    cid = new_correlation_id()
    limit = pipeline.scanner.max_input_bytes or DEFAULT_MAX_INPUT_BYTES
    try:
        # Check the size before reading, so an oversized file is never loaded into memory.
        too_large = path.stat().st_size > limit
        data = b"" if too_large else path.read_bytes()
    except OSError:
        _fail_item(index, cid, "input_unreadable")
        return True, None
    if too_large:
        _fail_item(index, cid, "input_too_large")
        return True, None

    digest = sha256_hex(data)
    if digest in seen:
        _emit(
            {
                "index": index,
                "duplicate_of": seen[digest],
                "processing_status": "skipped_duplicate",
            }
        )
        return False, None

    seen[digest] = cid
    result = pipeline.process(data, correlation_id=cid)
    try:
        _write_private(out_dir / f"{cid}.json", result.model_dump_json(indent=2))
        _append_private(
            manifest,
            json.dumps(
                {
                    "index": index,
                    "input_path": str(path),
                    "correlation_id": cid,
                    "source_sha256": digest,
                }
            ),
        )
    except OSError:  # disk full, permissions, ...: report it and carry on with the batch
        _fail_item(index, cid, "output_write_failed")
        return True, digest
    failed = result.processing_status is ProcessingStatus.FAILED
    _emit({"index": index, **result.safe_summary()})
    return failed, digest


def main(argv: list[str] | None = None, *, pipeline: DocumentPipeline | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:  # argparse exits on --help (0) and usage errors (2)
        return exc.code if isinstance(exc.code, int) else 2

    owns_pipeline = pipeline is None
    if pipeline is None:
        if args.env_file is not None and not Path(args.env_file).is_file():
            sys.stderr.write("configuration error: the --env-file given does not exist\n")
            return 2
        env_file = args.env_file or (".env" if Path(".env").is_file() else None)
        for name in unknown_setting_names(env_file):
            sys.stderr.write(f"configuration warning: {name} is not a recognised setting and is ignored\n")
        try:
            settings = load_settings(env_file)
        except PipelineError as exc:
            sys.stderr.write(f"configuration error: {exc.detail}\n")
            return 2
        configure_logging(settings.log_level)
        pipeline = build_pipeline(settings, use_llm=settings.enable_llm and not args.no_llm)

    try:
        out_dir: Path = args.out_dir
        try:
            out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError:
            sys.stderr.write("error: cannot create --out-dir\n")
            return 2
        manifest = out_dir / "manifest.jsonl"
        seen: dict[str, str] = {}
        any_failed = False

        for index, path in enumerate(expand_inputs(args.inputs)):
            failed, _ = _process_item(index, path, pipeline, out_dir, manifest, seen)
            if failed:
                any_failed = True

        return 1 if any_failed else 0
    finally:
        if owns_pipeline:
            pipeline.close()


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
