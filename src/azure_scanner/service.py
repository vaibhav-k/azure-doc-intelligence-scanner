from __future__ import annotations

import os

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential


class DocumentScannerService:
    """Handles communication with the Azure Document Intelligence API."""

    def __init__(
        self, endpoint: str, key: str, model_id: str = "prebuilt-read"
    ) -> None:
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )
        self.model_id = model_id

    def analyze_document(self, file_path: str):
        """
        Reads a local file and sends it to Azure for analysis using the configured model.

        Args:
            file_path (str): The path to the local file to be analyzed.

        Raises:
            FileNotFoundError: If the specified file does not exist.

        Returns:
            The result of the analysis from Azure.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at '{file_path}'.")

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        print(f"Analyzing '{file_path}' with model '{self.model_id}'...")

        poller = self.client.begin_analyze_document(
            model_id=self.model_id,
            body=AnalyzeDocumentRequest(bytes_source=file_bytes),
        )
        return poller.result()
