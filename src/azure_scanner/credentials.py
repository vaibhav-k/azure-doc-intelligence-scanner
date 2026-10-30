"""
Microsoft Entra ID credential selection, shared by every Azure client in one process.

* One credential instance is created per process and passed to both the Document Intelligence and
  Azure OpenAI clients, so tokens are cached once instead of per client.
* ``ENVIRONMENT=prod`` uses ``ManagedIdentityCredential`` only. Microsoft recommends replacing
  ``DefaultAzureCredential`` with a specific credential once an app is deployed: the chain is slower,
  harder to debug, and its behaviour changes if someone sets environment variables on the host.
  Set ``AZURE_CLIENT_ID`` to select a user-assigned managed identity; leave it unset for the
  system-assigned identity.
* ``dev`` and ``test`` use ``DefaultAzureCredential`` so a developer's ``az login`` works locally.
* API keys, when configured, bypass this module entirely (and are refused when ``ENVIRONMENT=prod``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

    from azure_scanner.config import Settings


def needs_entra_credential(settings: Settings, *, use_llm: bool) -> bool:
    """True when at least one configured service authenticates with Entra ID rather than a key."""
    di_keyless = settings.azure_document_intelligence_key is None
    llm_keyless = use_llm and settings.azure_openai_api_key is None
    return di_keyless or llm_keyless


def build_credential(settings: Settings) -> TokenCredential:
    if settings.environment == "prod":
        from azure.identity import ManagedIdentityCredential

        if settings.azure_client_id:
            return ManagedIdentityCredential(client_id=settings.azure_client_id)
        return ManagedIdentityCredential()

    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential()
