from azure.ai.documentintelligence.models import AnalyzeResult


def print_extracted_text(result: AnalyzeResult) -> None:
    """Formats and prints the extracted text lines from the analysis result."""
    print("\n--- Extracted Text ---")
    if not result.pages:
        print("No text detected in the document.")
        return

    for page in result.pages:
        print(f"\n--- Page {page.page_number} ---")
        if page.lines:
            for line in page.lines:
                print(line.content)
