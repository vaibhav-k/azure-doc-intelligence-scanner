import os

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, AnalyzeResult
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from dotenv import load_dotenv

# Load environment variables from a local .env file
load_dotenv()


def validate_environment(file_path: str):
    """
    Validate that the required environment variables are set and the file exists.

    Args:
        file_path (str): The path to the file to be validated.

    Raises:
        ValueError: If the required environment variables are not set or the file does not exist.
        FileNotFoundError: If the specified file does not exist.

    Returns:
        tuple: A tuple containing the endpoint and key.
    """
    endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    if not endpoint or not key:
        raise ValueError("Azure Document Intelligence endpoint and key must be set.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at '{file_path}'.")

    return endpoint, key


def print_extracted_text(result: AnalyzeResult):
    """
    Print the extracted text from the analyzed document result.

    Args:
        result (AnalyzeResult): The result of the document analysis.
    """
    if not result.pages:
        print("No text detected in the document.")
        return

    print("\n--- Extracted Text ---")
    for page in result.pages:
        print(f"\n--- Page {page.page_number} ---")
        if page.lines:
            for line in page.lines:
                print(line.content)


def scan_doctor_notes(file_path: str):
    """
    Scan the doctor's notes from the specified file path using Azure Document Intelligence.

    Args:
        file_path (str): The path to the doctor's notes file.

    Raises:
        AzureError: If an error occurs while calling the Azure Document Intelligence API.
        OSError: If an error occurs while reading the file.
    """
    endpoint, key = validate_environment(file_path)

    try:
        client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        print(f"Analyzing '{file_path}' with Prebuilt-Read model...")

        poller = client.begin_analyze_document(
            model_id="prebuilt-read",
            body=AnalyzeDocumentRequest(bytes_source=file_bytes),
        )
        result = poller.result()

        print_extracted_text(result)

    except AzureError as e:
        raise AzureError(f"Azure Document Intelligence API error occurred: {e}") from e
    except (OSError, ValueError) as e:
        raise OSError(f"File or configuration error occurred: {e}") from e


if __name__ == "__main__":
    PATH_TO_NOTE = "inputs/[clean] handwritten doctor's notes.jpeg"
    scan_doctor_notes(PATH_TO_NOTE)
