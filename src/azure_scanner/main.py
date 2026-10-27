import sys

from azure.core.exceptions import AzureError
from config import get_azure_config
from service import DocumentScannerService
from utils import print_extracted_text


def main():
    endpoint, key = get_azure_config()
    path_to_note = "inputs/[clean] handwritten doctor's notes.jpeg"

    try:
        scanner_service = DocumentScannerService(endpoint=endpoint, key=key)
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
