from __future__ import annotations

import json
import logging

import pytest

from azure_scanner.logging_utils import (
    QUIET_THIRD_PARTY_LOGGERS,
    SafeJsonFormatter,
    log_event,
    new_correlation_id,
)
from tests.conftest import PHI_SENTINELS


def test_disallowed_fields_are_dropped(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    log_event(
        logging.getLogger("t"),
        logging.INFO,
        "evt",
        correlation_id="c",
        patient_name=PHI_SENTINELS[0],
        ocr_text=PHI_SENTINELS[2],
        file_path=f"/scans/{PHI_SENTINELS[0]}.jpg",
    )
    record = caplog.records[-1]
    assert record.safe_fields == {"correlation_id": "c", "dropped_fields": 3}  # type: ignore[attr-defined]
    rendered = SafeJsonFormatter().format(record)
    assert all(s not in rendered for s in PHI_SENTINELS)


def test_formatter_never_renders_exception_message_or_traceback() -> None:
    try:
        raise ValueError(f"payload contained {PHI_SENTINELS[1]}")
    except ValueError:
        import sys

        record = logging.LogRecord("t", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())
    rendered = json.loads(SafeJsonFormatter().format(record))
    assert rendered["error_type"] == "ValueError"
    assert PHI_SENTINELS[1] not in json.dumps(rendered)


def test_correlation_ids_are_random_and_opaque() -> None:
    a, b = new_correlation_id(), new_correlation_id()
    assert a != b and len(a) == 32 and a.isalnum()


def test_openai_sdk_payload_debug_logs_are_blocked_regardless_of_app_config(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression: openai 1.106 logs request options (incl. OCR text) at DEBUG."""
    import azure_scanner.llm_parser  # noqa: F401  (installs the guard on import)

    caplog.set_level(logging.DEBUG)
    sdk = logging.getLogger("openai._base_client")
    sdk.setLevel(logging.DEBUG)
    sdk.debug("Request options: %s", {"messages": [{"content": PHI_SENTINELS[0]}]})
    sdk.warning("Retrying request")
    rendered = " ".join(r.getMessage() for r in caplog.records)
    assert PHI_SENTINELS[0] not in rendered
    assert "Retrying request" in rendered  # warnings still flow


def test_configure_logging_installs_safe_formatter() -> None:
    from azure_scanner.logging_utils import configure_logging

    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    try:
        configure_logging("WARNING")
        assert isinstance(root.handlers[0].formatter, SafeJsonFormatter)
        assert root.level == logging.WARNING
        assert logging.getLogger("openai").level == logging.WARNING
    finally:
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])


@pytest.mark.parametrize("name", ["httpx", "httpx2", "httpcore", "httpcore2", "urllib3", "azure"])
def test_configure_logging_silences_http_client_info_logs(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression: openai 3.x logs 'HTTP Request: ...' at INFO via ``httpx2``, which was not silenced."""
    from azure_scanner.logging_utils import configure_logging

    assert name in QUIET_THIRD_PARTY_LOGGERS
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    try:
        configure_logging("INFO")
        logging.getLogger(name).info("HTTP Request: %s %s", "POST", "https://example.invalid/x")
        logging.getLogger(name).warning("still visible")
        err = capsys.readouterr().err
        assert "HTTP Request" not in err
        assert "still visible" in err
    finally:
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])
        for n in QUIET_THIRD_PARTY_LOGGERS:
            logging.getLogger(n).setLevel(logging.NOTSET)
