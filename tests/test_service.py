from unittest.mock import MagicMock, mock_open, patch

import pytest

from azure_scanner.service import DocumentScannerService


@patch("azure_scanner.service.DocumentIntelligenceClient")
@patch("builtins.open", new_callable=mock_open, read_data=b"fake-image-bytes")
@patch("os.path.exists", return_value=True)
def test_analyze_document_success(mock_client_cls: MagicMock):
    mock_client = mock_client_cls.return_value
    mock_poller = MagicMock()
    mock_result = MagicMock()
    mock_poller.result.return_value = mock_result
    mock_client.begin_analyze_document.return_value = mock_poller

    service = DocumentScannerService("https://test.endpoint", "fake-key")
    result = service.analyze_document("dummy_note.jpg")

    assert result == mock_result
    mock_client.begin_analyze_document.assert_called_once()


def test_analyze_document_file_not_found():
    service = DocumentScannerService("https://test.endpoint", "fake-key")
    with pytest.raises(FileNotFoundError):
        service.analyze_document("nonexistent_note.jpg")
