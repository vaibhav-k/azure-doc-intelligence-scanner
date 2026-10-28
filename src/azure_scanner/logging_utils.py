"""Privacy-safe structured logging.

Policy: logs carry operational metadata only. ``log_event`` accepts an allow-list of field
names; anything else is dropped (and counted) rather than emitted. The JSON formatter never
renders exception messages or tracebacks - only the exception class name - because SDK
exception text can include request or response fragments. Correlation IDs are random and
contain no patient or document identifiers.

Removing console output does not by itself make a system compliant with any healthcare
privacy regime; it is one control among many.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

ALLOWED_FIELDS = frozenset(
    {
        "correlation_id",
        "stage",
        "status",
        "processing_status",
        "review_status",
        "error_code",
        "error_type",
        "retryable",
        "http_status",
        "duration_ms",
        "elapsed_ms",
        "attempt",
        "page_count",
        "scored_word_count",
        "flag_counts",
        "reason_count",
        "reason_codes",
        "input_format",
        "input_bytes",
        "preprocess_mode",
        "transformations",
        "deskew_reason",
        "model_id",
        "deployment",
        "prompt_version",
        "schema_version",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "finish_reason",
        "medication_count",
        "field_count",
        "from_status",
        "to_status",
        "actor_type",
        "dropped_fields",
        "pipeline_version",
    }
)

_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "message",
    "asctime",
}


def new_correlation_id() -> str:
    """Random, opaque identifier. Never derive it from patient or document identifiers."""
    return uuid.uuid4().hex


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    safe = {k: v for k, v in fields.items() if k in ALLOWED_FIELDS}
    dropped = len(fields) - len(safe)
    if dropped:
        safe["dropped_fields"] = dropped
    logger.log(level, event, extra={"safe_fields": safe})


class SafeJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": (record.msg if isinstance(record.msg, str) else type(record.msg).__name__),
        }
        payload.update(getattr(record, "safe_fields", {}))
        if record.exc_info and record.exc_info[0] is not None:
            payload["error_type"] = record.exc_info[0].__name__  # class name only, never the message
        return json.dumps(payload, default=str, sort_keys=True)


class _BelowWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING


_SDK_PAYLOAD_LOGGERS = ("openai", "openai._base_client")
_guard = _BelowWarningFilter()


def install_sdk_log_guards() -> None:
    """Drop DEBUG/INFO records from SDK loggers known to log request payloads.

    Verified: openai 1.106 logs the full request options - including message content, i.e. OCR
    text - at DEBUG via ``openai._base_client``. A logger-level filter applies regardless of how
    the host application configures handlers or levels (root DEBUG, ``OPENAI_LOG=debug``, ...).
    Idempotent. Called on import by ``llm_parser``.
    """
    for name in _SDK_PAYLOAD_LOGGERS:
        lg = logging.getLogger(name)
        if _guard not in lg.filters:
            lg.addFilter(_guard)


def configure_logging(level: str = "INFO") -> None:
    """Install the safe formatter on the root logger. Call once from the entry point."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(SafeJsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # Third-party HTTP loggers can emit URLs and headers at DEBUG; keep them quiet.
    for noisy in (
        "azure",
        "azure.core.pipeline.policies.http_logging_policy",
        "httpx",
        "openai",
        "urllib3",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
