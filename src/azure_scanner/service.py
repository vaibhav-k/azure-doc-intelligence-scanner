import os

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential


class DocumentScannerService:
    """Handles communication with the Azure Document Intelligence API."""

    def __init__(self, endpoint: str, key: str):
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

    def analyze_document(self, file_path: str):
        """Reads a local file and sends it to Azure for Prebuilt-Read analysis."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at '{file_path}'.")

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        print(f"Analyzing '{file_path}' with Prebuilt-Read model...")

        poller = self.client.begin_analyze_document(
            model_id="prebuilt-read",
            body=AnalyzeDocumentRequest(bytes_source=file_bytes),
        )
        return poller.result()
