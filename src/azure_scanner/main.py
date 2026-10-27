from __future__ import annotations

import sys

from azure.core.exceptions import AzureError

from src.azure_scanner.confidence import evaluate_ocr_confidence
from src.azure_scanner.config import get_azure_config
from src.azure_scanner.llm_parser import parse_markdown_with_gpt
from src.azure_scanner.service import DocumentScannerService


def main() -> None:
    endpoint, key, model_id, enable_llm, openai_deployment = get_azure_config()
    path_to_note = "./inputs/[clean] handwritten doctor's notes.jpeg"
    confidence_threshold = 0.85

    try:
        # Step 1: Analyze Document via Azure Document Intelligence
        scanner_service = DocumentScannerService(
            endpoint=endpoint, key=key, model_id=model_id
        )
        result = scanner_service.analyze_document(path_to_note)

        markdown_content = result.content if result.content else ""
        print("\n--- Extracted Markdown Content ---")
        print(
            markdown_content[:500] + "..."
            if len(markdown_content) > 500
            else markdown_content
        )

        # Step 2: Confidence Filtering & Human Review Queue Routing
        print(
            f"\nEvaluating word-level confidence scores (Threshold: {confidence_threshold})..."
        )
        _, review_queue = evaluate_ocr_confidence(
            result, threshold=confidence_threshold
        )

        if review_queue:
            print(
                f"\n⚠️ Routed {len(review_queue)} low-confidence token(s) to Human Review Queue:"
            )
            for item in review_queue:
                print(
                    f"  - Page {item['page']} | Token: '{item['token']}' | Confidence: {item['confidence']}"
                )
        else:
            print("\n✅ All extracted tokens met the high-confidence threshold.")

        # Step 3: Conditionally run LLM Structured Extraction
        if enable_llm and openai_deployment:
            print(
                f"\nParsing extracted markdown with Azure OpenAI (Deployment: {openai_deployment})..."
            )
            prescription_data = parse_markdown_with_gpt(
                markdown_content, deployment_name=openai_deployment
            )

            print("\n--- Structured Prescription Output ---")
            print(prescription_data.model_dump_json(indent=2))
        else:
            print("\n[INFO] LLM processing is disabled (ENABLE_LLM=false).")

    except AzureError as e:
        print(f"Azure API error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError) as e:
        print(f"File or configuration error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
