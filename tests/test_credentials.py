"""Credential selection and sharing. No tokens are requested: credentials are only constructed."""

from __future__ import annotations

import pytest

from azure_scanner.config import Settings
from azure_scanner.credentials import build_credential, needs_entra_credential

DI = "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    monkeypatch.setenv(DI, "https://unit-test.invalid/")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return Settings()  # type: ignore[call-arg]


def test_prod_uses_managed_identity_only(monkeypatch: pytest.MonkeyPatch) -> None:
    cred = build_credential(_settings(monkeypatch, ENVIRONMENT="prod"))
    assert type(cred).__name__ == "ManagedIdentityCredential"


def test_prod_accepts_a_user_assigned_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    client_id = "00000000-0000-0000-0000-000000000001"
    settings = _settings(monkeypatch, ENVIRONMENT="prod", AZURE_CLIENT_ID=client_id)
    assert settings.azure_client_id == client_id
    assert type(build_credential(settings)).__name__ == "ManagedIdentityCredential"


def test_dev_uses_the_developer_friendly_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    assert type(build_credential(_settings(monkeypatch))).__name__ == "DefaultAzureCredential"


@pytest.mark.parametrize(
    "env, use_llm, expected",
    [
        ({}, False, True),  # DI keyless
        ({"AZURE_DOCUMENT_INTELLIGENCE_KEY": "k"}, False, False),
        ({"AZURE_DOCUMENT_INTELLIGENCE_KEY": "k"}, True, True),  # OpenAI keyless
        (
            {"AZURE_DOCUMENT_INTELLIGENCE_KEY": "k", "AZURE_OPENAI_API_KEY": "k"},
            True,
            False,
        ),
    ],
)
def test_credential_only_needed_for_keyless_services(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], use_llm: bool, expected: bool
) -> None:
    assert needs_entra_credential(_settings(monkeypatch, **env), use_llm=use_llm) is expected


def test_one_credential_is_shared_by_both_clients_and_closed_with_the_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from azure.core.credentials import AccessToken

    import azure_scanner.main as cli

    class FakeCredential:
        closed = False

        def get_token(self, *scopes: str, **kwargs: object) -> AccessToken:
            raise AssertionError("no token should be requested during construction")

        def close(self) -> None:
            self.closed = True

    created: list[FakeCredential] = []

    def fake_build(settings: Settings) -> FakeCredential:
        created.append(FakeCredential())
        return created[-1]

    monkeypatch.setattr(cli, "build_credential", fake_build)
    settings = _settings(
        monkeypatch,
        ENABLE_LLM="true",
        AZURE_OPENAI_ENDPOINT="https://unit-test.invalid/",
        AZURE_OPENAI_DEPLOYMENT_NAME="d",
    )
    pipeline = cli.build_pipeline(settings, use_llm=True)
    assert len(created) == 1
    assert pipeline.scanner.client._config.credential is created[0]
    pipeline.close()
    assert created[0].closed


def test_no_credential_is_built_when_keys_are_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import azure_scanner.main as cli

    monkeypatch.setattr(cli, "build_credential", lambda s: pytest.fail("credential built"))
    settings = _settings(monkeypatch, AZURE_DOCUMENT_INTELLIGENCE_KEY="k")
    cli.build_pipeline(settings, use_llm=False).close()
