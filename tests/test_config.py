from __future__ import annotations

from pathlib import Path

import pytest

from azure_scanner.config import Settings, load_settings
from azure_scanner.errors import ConfigurationError
from azure_scanner.preprocessing import PreprocessMode

DI = "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"


def test_minimal_valid_settings_and_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DI, "https://test.cognitiveservices.azure.com/")
    s = load_settings(env_file=None)
    assert s.azure_document_intelligence_model == "prebuilt-layout"
    assert s.azure_document_intelligence_key is None  # -> Entra ID / managed identity
    assert s.enable_llm is False
    assert s.preprocess_mode is PreprocessMode.MINIMAL
    assert s.ocr_confidence_threshold == 0.85
    assert s.llm_temperature is None  # not sent unless configured


def test_existing_env_variable_names_still_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DI, "https://x.cognitiveservices.azure.com/")
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "fake-key")
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_MODEL", "prebuilt-read")
    monkeypatch.setenv("ENABLE_LLM", "true")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-chat-latest")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-10-01-preview")
    s = load_settings(env_file=None)
    assert s.azure_document_intelligence_key is not None
    assert s.azure_document_intelligence_key.get_secret_value() == "fake-key"
    assert s.azure_openai_api_version == "2024-10-01-preview"


def test_missing_endpoint_raises_typed_error_not_system_exit() -> None:
    with pytest.raises(ConfigurationError) as exc:
        load_settings(env_file=None)
    assert DI in (exc.value.detail or "")


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"ENABLE_LLM": "true"}, "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME"),
        ({"ENABLE_LLM": "true", "AZURE_OPENAI_ENDPOINT": "https://x/"}, "AZURE_OPENAI_DEPLOYMENT_NAME"),
        ({"ENVIRONMENT": "prod", "AZURE_DOCUMENT_INTELLIGENCE_KEY": "k"}, "managed identity"),
        ({"OCR_CONFIDENCE_THRESHOLD": "0"}, "OCR_CONFIDENCE_THRESHOLD"),
        ({"OCR_CONFIDENCE_THRESHOLD": "1.5"}, "OCR_CONFIDENCE_THRESHOLD"),
        ({"PREPROCESS_MODE": "aggressive"}, "PREPROCESS_MODE"),
        ({"LLM_TEMPERATURE": "-1"}, "LLM_TEMPERATURE"),
    ],
)
def test_invalid_combinations(monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected: str) -> None:
    monkeypatch.setenv(DI, "https://x/")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ConfigurationError) as exc:
        load_settings(env_file=None)
    assert expected in (exc.value.detail or "")


def test_http_endpoint_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DI, "http://insecure/")
    with pytest.raises(ConfigurationError, match="invalid_configuration"):
        load_settings(env_file=None)


def test_secrets_never_in_repr_or_error_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "super-secret-key-value-123"
    monkeypatch.setenv(DI, "https://x/")
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", secret)
    assert secret not in repr(load_settings(env_file=None))
    monkeypatch.setenv("ENVIRONMENT", "prod")
    with pytest.raises(ConfigurationError) as exc:
        load_settings(env_file=None)
    assert secret not in str(exc.value) and secret not in (exc.value.detail or "")


def test_env_file_is_read_only_when_requested(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = tmp_path / "custom.env"
    env.write_text(f"{DI}=https://from-file/\n")
    assert load_settings(env_file=str(env)).azure_document_intelligence_endpoint == "https://from-file/"
    with pytest.raises(ConfigurationError):
        load_settings(env_file=None)


def test_importing_config_has_no_side_effects(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(f"{DI}=https://should-not-load/\n")
    import importlib

    import azure_scanner.config as cfg

    importlib.reload(cfg)
    import os

    assert DI not in os.environ


def test_settings_is_immutable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DI, "https://x/")
    s = Settings()  # type: ignore[call-arg]
    with pytest.raises(Exception):  # noqa: B017 - pydantic frozen instance error
        s.environment = "prod"  # type: ignore[misc]
