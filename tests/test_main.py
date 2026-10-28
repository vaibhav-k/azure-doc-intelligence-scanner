from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from azure_scanner.main import main
from tests.conftest import PHI_SENTINELS
from tests.test_pipeline import _phi_extraction_json, _phi_ocr, _pipeline
from tests.test_llm_parser import Recorder, ok


@pytest.fixture(autouse=True)
def _no_sdk_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    monkeypatch.setattr(openai._base_client.time, "sleep", lambda s: None)


def _lines(out: str) -> list[dict]:  # type: ignore[type-arg]
    return [json.loads(line) for line in out.strip().splitlines()]


def test_cli_writes_private_results_and_prints_only_safe_summary(
    tmp_path: Path, page_png: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / f"{PHI_SENTINELS[0]}_scan.png"  # PHI in the file name, as often happens
    src.write_bytes(page_png)
    out_dir = tmp_path / "out"
    pipe, _ = _pipeline(_phi_ocr(), Recorder([ok(_phi_extraction_json())]))

    code = main(["process", str(src), "--out-dir", str(out_dir)], pipeline=pipe)
    stdout = capsys.readouterr()
    assert code == 0
    (line,) = _lines(stdout.out)
    assert line["review_status"] == "needs_review" and line["processing_status"] == "completed"
    for s in PHI_SENTINELS:
        assert s not in stdout.out and s not in stdout.err

    result_file = out_dir / f"{line['correlation_id']}.json"
    assert result_file.exists() and PHI_SENTINELS[0] not in result_file.name
    assert PHI_SENTINELS[2] in result_file.read_text()  # full result kept, but only on disk
    if os.name == "posix":
        assert stat.S_IMODE(result_file.stat().st_mode) == 0o600
        assert stat.S_IMODE((out_dir / "manifest.jsonl").stat().st_mode) == 0o600
    manifest = _lines((out_dir / "manifest.jsonl").read_text())
    assert manifest[0]["correlation_id"] == line["correlation_id"]


def test_cli_duplicate_inputs_processed_once(tmp_path: Path, page_png: bytes, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(page_png)
    b.write_bytes(page_png)
    pipe, di = _pipeline()
    assert main(["process", str(a), str(b), "--out-dir", str(tmp_path / "o")], pipeline=pipe) == 0
    lines = _lines(capsys.readouterr().out)
    assert lines[1]["processing_status"] == "skipped_duplicate"
    assert lines[1]["duplicate_of"] == lines[0]["correlation_id"]
    assert di.begin_analyze_document.call_count == 1


def test_cli_exit_code_1_when_any_input_fails(tmp_path: Path, page_png: bytes, capsys: pytest.CaptureFixture[str]) -> None:
    good, bad = tmp_path / "good.png", tmp_path / "bad.bin"
    good.write_bytes(page_png)
    bad.write_bytes(b"not an image")
    missing = tmp_path / "missing.png"
    pipe, _ = _pipeline()
    code = main(["process", str(good), str(bad), str(missing), "--out-dir", str(tmp_path / "o")], pipeline=pipe)
    lines = _lines(capsys.readouterr().out)
    assert code == 1
    assert [ln["processing_status"] for ln in lines] == ["completed", "failed", "failed"]
    assert lines[1]["error_code"] == "unsupported_format" and lines[2]["error_code"] == "input_unreadable"


def test_cli_configuration_error_exit_2_names_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["process", "x.png", "--out-dir", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert code == 2 and "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT" in err


def test_cli_usage_errors() -> None:
    assert main([]) == 2
    assert main(["--help"]) == 0
