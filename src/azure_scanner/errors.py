"""Typed, PHI-free exceptions shared across pipeline stages.

Every error carries a stable machine-readable ``code`` and the ``stage`` that raised it.
``str(error)`` is the code only, so logging an exception can never echo document text,
request payloads, credentials or patient identifiers. Underlying SDK exceptions are kept
as ``__cause__`` for local debugging but must not be logged with their message.
"""

from __future__ import annotations


class PipelineError(Exception):
    stage: str = "unknown"
    retryable: bool = False

    def __init__(self, code: str, *, retryable: bool | None = None, detail: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        if retryable is not None:
            self.retryable = retryable
        # ``detail`` must only ever hold PHI-free operational data (e.g. an HTTP status or a
        # request-parameter NAME). Never pass exception messages or content here.
        self.detail = detail

    def __str__(self) -> str:
        return self.code


class ConfigurationError(PipelineError):
    stage = "configuration"


class OcrError(PipelineError):
    stage = "ocr"
