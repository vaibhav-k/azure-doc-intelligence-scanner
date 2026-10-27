# Azure Document Intelligence Note Scanner

A lightweight, robust Python script to extract handwritten and printed text from medical notes, scanned prescriptions, or documents using the modern Azure Document Intelligence SDK.

## Prerequisites

* Python 3.8+

* An active **Azure Document Intelligence resource** in your Azure portal.

## Setup Instructions

1. Clone or download this repository:
   ```powershell
   git clone https://github.com/vaibhav-k/azure-doc-intelligence-scanner.git
   cd azure-doc-intelligence-scanner
   ```

2. Create and activate a virtual environment:

   1. Windows (PowerShell):

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

   2. macOS / Linux:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

4. Configure your credentials:
   - Copy the template file: `cp .env.example .env`

   - Open .env and fill out your configuration:

   ```powershell
   AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource-name.cognitiveservices.azure.com
   AZURE_DOCUMENT_INTELLIGENCE_KEY=your-api-key-here
   AZURE_DOCUMENT_INTELLIGENCE_MODEL=prebuilt-layout
   ```

## Model Options

You can configure which model the scanner invokes by updating the AZURE_DOCUMENT_INTELLIGENCE_MODEL variable in your .env file:

* `prebuilt-read`: • High-resolution OCR optimized for raw text lines and messy handwriting blocks.

* `prebuilt-layout` (Recommended): Extracts text alongside document structure, tables, and selection marks.

*Note: Recommended if you plan to pipe markdown data downstream into an LLM (like GPT-4o) to safely parse complex handwritten prescriptions.*

* Custom Models: Use your unique custom model ID if you have trained a domain-specific model using Document Intelligence Studio.

## Running the Script

1. Update the `PATH_TO_NOTE` variable in `main.py` with the path to your target image or PDF.

2. Run the script:

```
python main.py
```

## Code Quality & Linting

This project uses `Black` for code formatting and `Ruff` for linting and code checks to maintain high code standards.

- Format code: `python -m black .`

- Linter check and fix: `ruff check . --fix`
