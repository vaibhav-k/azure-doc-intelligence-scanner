from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def get_azure_config() -> tuple[str, str, str, bool, str | None]:
    """Retrieves and validates Azure credentials, model ID, the LLM toggle, and the OpenAI deployment name."""
    endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    model_id = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_MODEL", "prebuilt-layout")

    enable_llm_str = os.environ.get("ENABLE_LLM", "false").strip().lower()
    enable_llm = enable_llm_str in ("true", "1", "t", "yes")

    openai_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

    if not endpoint or not key:
        print(
            "Error: Azure Document Intelligence endpoint and key must be set in environment variables or a .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    if enable_llm and not openai_deployment:
        print(
            "Error: AZURE_OPENAI_DEPLOYMENT_NAME must be set when ENABLE_LLM=true.",
            file=sys.stderr,
        )
        sys.exit(1)

    return endpoint, key, model_id, enable_llm, openai_deployment
