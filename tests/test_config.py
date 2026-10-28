import pytest

from src.azure_scanner.config import get_azure_config


def test_get_azure_config_success(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        "https://test.cognitiveservices.azure.com/",
    )
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "fake-key")

    endpoint, key = get_azure_config()
    assert endpoint == "https://test.cognitiveservices.azure.com/"
    assert key == "fake-key"


def test_get_azure_config_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        get_azure_config()
    assert excinfo.value.code == 1
