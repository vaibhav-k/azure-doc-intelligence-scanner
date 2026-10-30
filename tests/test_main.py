from __future__ import annotations

import json
from pathlib import Path

import pytest

from azure_scanner.main import main
from tests.conftest import PHI_SENTINELS, encode, synthetic_page
from tests.test_llm_parser import Recorder, ok
from tests.test_pipeline import _phi_extraction_json, _phi_ocr, _pipeline


@pytest.fixture(autouse=True)
def _no_sdk_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("openai._base_client.time.sleep", lambda _: None)


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
    assert line["review_status"] == "needs_review"
    assert line["processing_status"] == "completed"
    for s in PHI_SENTINELS:
        assert s not in stdout.out
        assert s not in stdout.err

    result_file = out_dir / f"{line['correlation_id']}.json"
    assert result_file.exists()
    assert PHI_SENTINELS[0] not in result_file.name
    assert PHI_SENTINELS[2] in result_file.read_text()  # full result kept, but only on disk
    manifest = _lines((out_dir / "manifest.jsonl").read_text())
    assert manifest[0]["correlation_id"] == line["correlation_id"]


def test_cli_duplicate_inputs_processed_once(
    tmp_path: Path, page_png: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(page_png)
    b.write_bytes(page_png)
    pipe, di = _pipeline()
    assert main(["process", str(a), str(b), "--out-dir", str(tmp_path / "o")], pipeline=pipe) == 0
    lines = _lines(capsys.readouterr().out)
    assert lines[1]["processing_status"] == "skipped_duplicate"
    assert lines[1]["duplicate_of"] == lines[0]["correlation_id"]
    assert di.begin_analyze_document.call_count == 1


def test_cli_exit_code_1_when_any_input_fails(
    tmp_path: Path, page_png: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    good, bad = tmp_path / "good.png", tmp_path / "bad.bin"
    good.write_bytes(page_png)
    bad.write_bytes(b"not an image")
    missing = tmp_path / "missing.png"
    pipe, _ = _pipeline()
    code = main(
        [
            "process",
            str(good),
            str(bad),
            str(missing),
            "--out-dir",
            str(tmp_path / "o"),
        ],
        pipeline=pipe,
    )
    lines = _lines(capsys.readouterr().out)
    assert code == 1
    assert [ln["processing_status"] for ln in lines] == [
        "completed",
        "failed",
        "failed",
    ]
    assert lines[1]["error_code"] == "unsupported_format"
    assert lines[2]["error_code"] == "input_unreadable"


def test_cli_configuration_error_exit_2_names_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["process", "x.png", "--out-dir", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert code == 2
    assert "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT" in err


def test_cli_usage_errors() -> None:
    assert main([]) == 2
    assert main(["--help"]) == 0


# ---------- input expansion, limits, write failures, configuration, cleanup ----------


def test_folder_and_wildcard_inputs_are_expanded(
    tmp_path: Path, page_png: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    from azure_scanner.main import expand_inputs

    scans = tmp_path / "scans"
    scans.mkdir()
    for name in ("a.png", "b.png", ".hidden.png"):
        (scans / name).write_bytes(page_png)
    bracketed = tmp_path / "[clean] note.png"  # glob characters in a real file name stay literal
    bracketed.write_bytes(page_png)

    assert [p.name for p in expand_inputs([scans])] == ["a.png", "b.png"]
    assert [p.name for p in expand_inputs([scans / "*.png"])] == ["a.png", "b.png"]
    assert expand_inputs([bracketed]) == [bracketed]
    assert expand_inputs([tmp_path / "nothing*.png"]) == [tmp_path / "nothing*.png"]  # fails later

    pipe, di = _pipeline()
    assert (
        main(
            ["process", str(scans / "*.png"), "--out-dir", str(tmp_path / "o")],
            pipeline=pipe,
        )
        == 0
    )
    assert len(_lines(capsys.readouterr().out)) == 2


def test_oversized_input_is_rejected_without_reading_it(
    tmp_path: Path,
    page_png: bytes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    big = tmp_path / "big.png"
    big.write_bytes(page_png)
    pipe, di = _pipeline()
    pipe.scanner.max_input_bytes = len(page_png) - 1

    def must_not_read(self: Path) -> bytes:
        raise AssertionError("oversized input was read into memory")

    monkeypatch.setattr(Path, "read_bytes", must_not_read)
    assert main(["process", str(big), "--out-dir", str(tmp_path / "o")], pipeline=pipe) == 1
    (line,) = _lines(capsys.readouterr().out)
    assert line["error_code"] == "input_too_large"
    di.begin_analyze_document.assert_not_called()


def test_write_failure_is_reported_and_the_batch_continues(
    tmp_path: Path,
    page_png: bytes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import azure_scanner.main as cli

    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(page_png)
    b.write_bytes(encode(synthetic_page(text="Rx Drug-Y 1 mg")))  # a different, valid page
    real_write = cli._write_private
    calls = {"n": 0}

    def flaky_write(path: Path, text: str) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        real_write(path, text)

    monkeypatch.setattr(cli, "_write_private", flaky_write)
    pipe, _ = _pipeline()
    code = main(["process", str(a), str(b), "--out-dir", str(tmp_path / "o")], pipeline=pipe)
    lines = _lines(capsys.readouterr().out)
    assert code == 1
    assert lines[0]["error_code"] == "output_write_failed"
    assert lines[1]["processing_status"] == "completed"


def test_explicit_env_file_must_exist(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "process",
            "x.png",
            "--out-dir",
            str(tmp_path / "o"),
            "--env-file",
            "missing.env",
        ]
    )
    assert code == 2
    assert "--env-file" in capsys.readouterr().err


def test_mistyped_setting_names_are_warned_about_by_name_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OCR_CONFIDENCE_TRESHOLD", "0.42")
    (tmp_path / ".env").write_text("LLM_TEMPRATURE=0.9\nAZURE_CLIENT_ID=not-owned-prefix\n")
    code = main(["process", "x.png", "--out-dir", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert code == 2  # still fails on the missing endpoint
    assert "OCR_CONFIDENCE_TRESHOLD is not a recognised setting" in err
    assert "LLM_TEMPRATURE is not a recognised setting" in err
    assert "0.42" not in err and "0.9" not in err  # names only, never values
    assert "AZURE_CLIENT_ID" not in err


def test_cli_closes_the_pipeline_it_built(
    tmp_path: Path, page_png: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    import azure_scanner.main as cli

    src = tmp_path / "a.png"
    src.write_bytes(page_png)
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://unit-test.invalid/")
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "fake")
    pipe, _ = _pipeline()
    closed: list[bool] = []
    monkeypatch.setattr(pipe, "close", lambda: closed.append(True))
    monkeypatch.setattr(cli, "build_pipeline", lambda settings, use_llm: pipe)
    assert main(["process", str(src), "--out-dir", str(tmp_path / "o")]) == 0
    assert closed == [True]
