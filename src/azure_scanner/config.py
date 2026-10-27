from __future__ import annotations

import os

from dotenv import load_dotenv

# Load environment variables from a local .env file
load_dotenv()


def get_azure_config() -> tuple[str, str, str]:
    """
    Retrieves and validates Azure Document Intelligence credentials.

    Raises:
        ValueError: If any of the required environment variables are not set.

    Returns:
        tuple[str, str, str]: A tuple containing the endpoint, key, and model ID.
    """
    endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    model_id = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_MODEL", "prebuilt-read")

    if not endpoint or not key:
        raise ValueError(
            "Azure Document Intelligence endpoint and key must be set in environment variables or a .env file."
        )

    return endpoint, key, model_id
