from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def get_azure_config() -> tuple[str, str, str, bool]:
    """
    Retrieves and validates Azure credentials, model ID, and the LLM toggle.

    Raises:
        SystemExit: If the endpoint or key is not set.

    Returns:
        tuple[str, str, str, bool]: A tuple containing the endpoint, key, model ID, and LLM toggle.
    """
    endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    model_id = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_MODEL", "prebuilt-layout")

    enable_llm_str = os.environ.get("ENABLE_LLM", "false").strip().lower()
    enable_llm = enable_llm_str in ("true", "1", "t", "yes")

    if not endpoint or not key:
        print(
            "Error: Azure Document Intelligence endpoint and key must be set in environment variables or a .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    return endpoint, key, model_id, enable_llm
