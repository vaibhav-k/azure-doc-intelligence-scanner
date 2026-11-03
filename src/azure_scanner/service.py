"""
Azure Document Intelligence adapter with incremental non-blocking LRO polling.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ServiceRequestError,
    ServiceResponseError,
)

from azure_scanner.errors import OcrError
from azure_scanner.logging_utils import log_event
from azure_scanner.preprocessing import (
    PreprocessMode,
    PreprocessResult,
    preprocess_bytes,
)

if TYPE_CHECKING:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import TokenCredential

    from azure_scanner.config import Settings

logger = logging.getLogger(__name__)

_MARKDOWN_MODELS = frozenset({"prebuilt-layout"})


def build_di_client(
    settings: Settings, credential: TokenCredential | None = None
) -> DocumentIntelligenceClient:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    auth: Any
    if settings.azure_document_intelligence_key is not None:
        auth = AzureKeyCredential(settings.azure_document_intelligence_key.get_secret_value())
    else:
        from azure_scanner.credentials import build_credential

        auth = credential if credential is not None else build_credential(settings)
    return DocumentIntelligenceClient(
        endpoint=settings.azure_document_intelligence_endpoint,
        credential=auth,
        retry_total=settings.ocr_max_retries,
        retry_backoff_factor=0.8,
        retry_backoff_max=30,
        logging_enable=False,
    )


def _http_error(exc: HttpResponseError) -> OcrError:
    status = getattr(exc, "status_code", None)
    retryable = status in (408, 429, 500, 502, 503, 504)
    return OcrError(f"ocr_http_{status or 'error'}", retryable=retryable, detail=str(status))


class DocumentScannerService:
    def __init__(
        self,
        client: DocumentIntelligenceClient,
        model_id: str = "prebuilt-layout",
        *,
        preprocess_mode: PreprocessMode = PreprocessMode.MINIMAL,
        timeout_seconds: float = 120.0,
        polling_interval_seconds: float = 1.0,
        max_input_bytes: int | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.client = client
        self.model_id = model_id
        self.preprocess_mode = preprocess_mode
        self.timeout_seconds = timeout_seconds
        self.polling_interval_seconds = polling_interval_seconds
        self.max_input_bytes = max_input_bytes

    def prepare(self, data: bytes, *, correlation_id: str) -> PreprocessResult:
        kwargs = {"max_input_bytes": self.max_input_bytes} if self.max_input_bytes else {}
        prepared = preprocess_bytes(data, self.preprocess_mode, **kwargs)
        log_event(
            logger,
            logging.INFO,
            "input_prepared",
            correlation_id=correlation_id,
            stage="preprocess",
            input_format=prepared.input_format.value,
            input_bytes=len(data),
            preprocess_mode=prepared.mode.value,
            transformations=list(prepared.transformations),
            deskew_reason=prepared.deskew.reason if prepared.deskew else None,
        )
        return prepared

    def run_ocr(self, payload: bytes, *, correlation_id: str) -> Any:
        from azure.ai.documentintelligence.models import (
            AnalyzeDocumentRequest,
            DocumentContentFormat,
        )

        kwargs: dict[str, Any] = {}
        if self.model_id in _MARKDOWN_MODELS:
            kwargs["output_content_format"] = DocumentContentFormat.MARKDOWN
        started = time.monotonic()
        try:
            poller = self.client.begin_analyze_document(
                model_id=self.model_id,
                body=AnalyzeDocumentRequest(bytes_source=payload),
                **kwargs,
            )
            while not poller.done():
                if time.monotonic() - started > self.timeout_seconds:
                    raise OcrError("ocr_timeout", retryable=True)
                time.sleep(self.polling_interval_seconds)

            result = poller.result()
            if result is None:
                raise OcrError("ocr_empty_result")
        except OcrError:
            raise
        except ClientAuthenticationError as exc:
            raise OcrError("ocr_auth_failed") from exc
        except HttpResponseError as exc:
            raise _http_error(exc) from exc
        except (ServiceRequestError, ServiceResponseError) as exc:
            raise OcrError("ocr_connection_error", retryable=True) from exc

        log_event(
            logger,
            logging.INFO,
            "ocr_completed",
            correlation_id=correlation_id,
            stage="ocr",
            model_id=self.model_id,
            page_count=len(getattr(result, "pages", None) or []),
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
        return result

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def analyze_bytes(self, data: bytes, *, correlation_id: str) -> Any:
        return self.run_ocr(
            self.prepare(data, correlation_id=correlation_id).payload,
            correlation_id=correlation_id,
        )

    def analyze_document(self, file_path: str, *, correlation_id: str = "local") -> Any:
        with open(file_path, "rb") as fh:
            return self.analyze_bytes(fh.read(), correlation_id=correlation_id)
