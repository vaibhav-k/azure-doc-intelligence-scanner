from __future__ import annotations

import json
import logging

import pytest

from azure_scanner.logging_utils import SafeJsonFormatter, log_event, new_correlation_id
from tests.conftest import PHI_SENTINELS


def test_disallowed_fields_are_dropped(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    log_event(logging.getLogger("t"), logging.INFO, "evt", correlation_id="c", patient_name=PHI_SENTINELS[0],
              ocr_text=PHI_SENTINELS[2], file_path=f"/scans/{PHI_SENTINELS[0]}.jpg")
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
