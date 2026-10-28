from __future__ import annotations

import os

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    DocumentContentFormat,
)
from azure.core.credentials import AzureKeyCredential

from azure_scanner.preprocessing import preprocess_image


class DocumentScannerService:
    """Handles communication and image preprocessing for the Azure Document Intelligence API."""

    def __init__(
        self, endpoint: str, key: str, model_id: str = "prebuilt-layout"
    ) -> None:
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )
        self.model_id = model_id

    def analyze_document(self, file_path: str):
        """
        Preprocesses a local document, sends it to Azure, and returns the raw analysis result object.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at '{file_path}'.")

        print(
            f"Preprocessing '{file_path}' (deskewing, noise reduction, contrast enhancement)..."
        )
        file_bytes = preprocess_image(file_path)

        print(f"Analyzing document with model '{self.model_id}' (Markdown output)...")
        poller = self.client.begin_analyze_document(
            model_id=self.model_id,
            body=AnalyzeDocumentRequest(bytes_source=file_bytes),
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
        return poller.result()
