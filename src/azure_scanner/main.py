from __future__ import annotations

import sys

from azure.core.exceptions import AzureError

from src.azure_scanner.config import get_azure_config
from src.azure_scanner.service import DocumentScannerService
from src.azure_scanner.utils import print_extracted_text


def main() -> None:
    endpoint, key, model_id = get_azure_config()
    path_to_note = "./inputs/[clean] handwritten doctor's notes.jpeg"

    try:
        scanner_service = DocumentScannerService(
            endpoint=endpoint, key=key, model_id=model_id
        )
        result = scanner_service.analyze_document(path_to_note)
        print_extracted_text(result)

    except AzureError as e:
        print(
            f"Azure Document Intelligence API error occurred: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    except (OSError, ValueError) as e:
        print(f"File or configuration error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
