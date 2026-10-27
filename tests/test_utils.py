from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from azure_scanner.utils import print_extracted_text


def test_print_extracted_text_with_content(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_line1 = MagicMock()
    mock_line1.content = "Patient Diagnosis: Normal"

    mock_line2 = MagicMock()
    mock_line2.content = "Prescription: Rest"

    mock_page = MagicMock()
    mock_page.page_number = 1
    mock_page.lines = [mock_line1, mock_line2]

    mock_result = MagicMock()
    mock_result.pages = [mock_page]

    print_extracted_text(mock_result)
    captured = capsys.readouterr()

    assert "Page 1" in captured.out
    assert "Patient Diagnosis: Normal" in captured.out
    assert "Prescription: Rest" in captured.out


def test_print_extracted_text_empty(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_result = MagicMock()
    mock_result.pages = []

    print_extracted_text(mock_result)
    captured = capsys.readouterr()

    assert "No text detected in the document." in captured.out
