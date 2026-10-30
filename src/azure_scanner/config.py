"""
Typed, validated settings.

* No import-time side effects and no ``sys.exit``: the entry point calls ``load_settings()``
  and decides how to exit; library code receives a ``Settings`` instance.
* Environment variable names are unchanged from the previous version
  (``AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`` etc.), so existing ``.env`` files keep working.
* Keys are optional. If a key is absent, Microsoft Entra ID is used (see ``credentials.py``:
  a managed identity in prod, ``DefaultAzureCredential`` in dev/test). ``ENVIRONMENT=prod``
  refuses API keys.
* Validation errors list variable NAMES only, never values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from azure_scanner.confidence import DEFAULT_OCR_CONFIDENCE_THRESHOLD
from azure_scanner.errors import ConfigurationError
from azure_scanner.preprocessing import DEFAULT_MAX_INPUT_BYTES, PreprocessMode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore", frozen=True, case_sensitive=False)

    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Client ID of a user-assigned managed identity (prod). Unset = system-assigned identity.
    azure_client_id: str | None = None

    # --- Azure Document Intelligence ---
    azure_document_intelligence_endpoint: str
    azure_document_intelligence_key: SecretStr | None = None
    azure_document_intelligence_model: str = "prebuilt-layout"
    ocr_timeout_seconds: float = Field(default=120.0, gt=0, le=900)
    ocr_max_retries: int = Field(default=3, ge=0, le=6)
    ocr_confidence_threshold: float = Field(default=DEFAULT_OCR_CONFIDENCE_THRESHOLD, gt=0, le=1)

    # --- Preprocessing ---
    preprocess_mode: PreprocessMode = PreprocessMode.MINIMAL
    max_input_bytes: int = Field(default=DEFAULT_MAX_INPUT_BYTES, gt=0, le=500 * 1024 * 1024)

    # --- Azure OpenAI ---
    enable_llm: bool = False
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_deployment_name: str | None = None
    # Unset => the v1 GA surface (<endpoint>/openai/v1/). Set (e.g. 2024-10-01-preview) => the
    # legacy AzureOpenAI client with that api-version, preserving the previous behaviour.
    azure_openai_api_version: str | None = None
    azure_openai_token_scope: str | None = None  # override the Entra ID scope if your resource needs it
    # Unset => temperature is NOT sent (server default). Compatibility of explicit values with the
    # configured deployment is unverified - see README "Known limitations and unverified items".
    llm_temperature: float | None = Field(default=None, ge=0, le=2)
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_max_input_chars: int = Field(default=60_000, gt=0)
    # Upper bound on generated tokens (sent as ``max_completion_tokens``). Unset = not sent, which is
    # the configuration verified live; set it to cap cost and latency per document. Reasoning models
    # count their reasoning tokens against this limit, so leave generous headroom.
    llm_max_output_tokens: int | None = Field(default=None, gt=0, le=128_000)

    @field_validator("azure_document_intelligence_endpoint", "azure_openai_endpoint")
    @classmethod
    def _https(cls, v: str | None) -> str | None:
        if v is not None and v and not v.lower().startswith("https://"):
            raise ValueError("endpoint must use https://")
        return v or None

    @model_validator(mode="after")
    def _cross_field(self) -> Settings:
        if self.enable_llm:
            missing = [
                name
                for name, value in (
                    ("AZURE_OPENAI_ENDPOINT", self.azure_openai_endpoint),
                    ("AZURE_OPENAI_DEPLOYMENT_NAME", self.azure_openai_deployment_name),
                )
                if not value
            ]
            if missing:
                raise ValueError(f"ENABLE_LLM=true requires {', '.join(missing)}")
        if self.environment == "prod" and (self.azure_document_intelligence_key or self.azure_openai_api_key):
            raise ValueError("API keys are not permitted when ENVIRONMENT=prod; use managed identity")
        return self


def load_settings(env_file: str | None = ".env") -> Settings:
    """
    Load settings from the environment (and optionally a .env file).

    Raises ``ConfigurationError`` whose ``detail`` lists the offending variable names only.
    """
    try:
        return Settings(_env_file=env_file)  # type: ignore[call-arg]
    except ValidationError as exc:
        problems: list[str] = []
        for err in exc.errors(include_input=False, include_url=False):
            loc = ".".join(str(p) for p in err["loc"]).upper() or "SETTINGS"
            msg = str(err.get("msg", "invalid")).removeprefix("Value error, ")
            problems.append(f"{loc}: {msg}")
        raise ConfigurationError("invalid_configuration", detail="; ".join(problems)) from None


# Prefixes this application owns. Unrecognised names under them are almost always typos
# (e.g. OCR_CONFIDENCE_TRESHOLD), which pydantic-settings would otherwise ignore silently.
_OWNED_PREFIXES = (
    "AZURE_DOCUMENT_INTELLIGENCE_",
    "AZURE_OPENAI_",
    "OCR_",
    "LLM_",
    "PREPROCESS_",
)


def unknown_setting_names(env_file: str | None = None) -> list[str]:
    """Names (never values) of variables under this app's prefixes that no setting reads."""
    known = {name.upper() for name in Settings.model_fields}
    names = set(os.environ)
    if env_file and Path(env_file).is_file():
        from dotenv import dotenv_values

        names |= set(dotenv_values(env_file))
    return sorted(n for n in (x.upper() for x in names) if n.startswith(_OWNED_PREFIXES) and n not in known)
