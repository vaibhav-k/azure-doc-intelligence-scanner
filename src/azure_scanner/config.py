from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

# Load environment variables from a local .env file
load_dotenv()


def get_azure_config() -> tuple[str, str]:
    """Retrieves and validates Azure Document Intelligence credentials."""
    endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    if not endpoint or not key:
        print(
            "Error: Azure Document Intelligence endpoint and key must be set in environment variables or a .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    return endpoint, key
