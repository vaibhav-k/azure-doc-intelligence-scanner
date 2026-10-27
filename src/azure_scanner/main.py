from __future__ import annotations

import sys

from azure.core.exceptions import AzureError

from src.azure_scanner.config import get_azure_config
from src.azure_scanner.llm_parser import parse_markdown_with_gpt
from src.azure_scanner.service import DocumentScannerService


def main() -> None:
    endpoint, key, model_id, enable_llm = get_azure_config()
    path_to_note = "./inputs/[clean] handwritten doctor's notes.jpeg"

    try:
        # Step 1: Extract Markdown via Azure Document Intelligence
        scanner_service = DocumentScannerService(
            endpoint=endpoint, key=key, model_id=model_id
        )
        markdown_content = scanner_service.analyze_document_to_markdown(path_to_note)

        print("\n--- Extracted Markdown Content ---")
        print(markdown_content)

        # Step 2: Conditionally run LLM Structured Extraction
        if enable_llm:
            print("\nParsing extracted markdown with Azure OpenAI GPT-4o...")
            prescription_data = parse_markdown_with_gpt(markdown_content)

            print("\n--- Structured Prescription Output ---")
            print(prescription_data.model_dump_json(indent=2))
        else:
            print(
                "\n[INFO] LLM processing is disabled (ENABLE_LLM=false). Skipping Azure OpenAI extraction."
            )

    except AzureError as e:
        print(
            f"Azure API error occurred: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    except (OSError, ValueError) as e:
        print(f"File or configuration error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
