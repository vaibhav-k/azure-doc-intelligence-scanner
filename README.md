# Azure Document Intelligence Note Scanner

A robust, production-ready Python pipeline to extract handwritten clinical notes and prescriptions using **Azure Document Intelligence**, **OpenCV image preprocessing**, and **Azure OpenAI GPT-4o Structured Outputs** (powered by Pydantic v2).

## Key Features

- **Advanced Image Preprocessing:** Automatically deskews, applies bilateral noise reduction, and enhances contrast using CLAHE via OpenCV before hitting the API.
- **Markdown OCR Extraction:** Utilizes Azure Document Intelligence (`prebuilt-layout`) to export hierarchical document structure and text as Markdown.
- **Hybrid LLM Reasoning:** Conditionally pipes OCR Markdown into Azure OpenAI GPT-4o using structured outputs (`client.beta.chat.completions.parse`) to cleanly normalize medications, dosages, frequencies, and clinical notes.
- **LLM Toggle:** Easily switch between raw OCR Markdown extraction (`ENABLE_LLM=false`) and full AI-parsed structured data (`ENABLE_LLM=true`).

## Prerequisites

- Python 3.10+
- An active **Azure Document Intelligence resource** in your Azure portal.
- An active **Azure OpenAI resource** with a `gpt-4o` deployment (if `ENABLE_LLM=true`).

## Setup Instructions

1. Clone or download this repository:
   ```powershell
   git clone [https://github.com/vaibhav-k/azure-doc-intelligence-scanner.git](https://github.com/vaibhav-k/azure-doc-intelligence-scanner.git)
   cd azure-doc-intelligence-scanner
   ```

2. Create and activate a virtual environment:
* **Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

* **macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies and the package in editable mode:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

4. Configure your credentials and feature toggles:
* Copy the template file: `cp .env.example .env`
* Open `.env` and configure your keys and settings:
```env
# Azure Document Intelligence Configuration
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=[https://your-resource-name.cognitiveservices.azure.com/](https://your-resource-name.cognitiveservices.azure.com/)
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-doc-intelligence-key-here
AZURE_DOCUMENT_INTELLIGENCE_MODEL=prebuilt-layout

# LLM Extraction Toggle & Azure OpenAI Configuration
ENABLE_LLM=true

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT=[https://your-openai-resource.openai.azure.com/](https://your-openai-resource.openai.azure.com/)
AZURE_OPENAI_API_KEY=your-openai-api-key-here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-10-01-preview
```

## Model Options

### Document Intelligence Models (`AZURE_DOCUMENT_INTELLIGENCE_MODEL`)

* **`prebuilt-layout`** *(Recommended)*: Extracts text alongside document structure, tables, selection marks, and Markdown formatting. Ideal for downstream LLM ingestion.
* **`prebuilt-read`**: High-resolution OCR optimized for raw text lines and handwriting blocks.
* **Custom Models**: Use your unique custom model ID if you have trained a domain-specific model in Document Intelligence Studio.

## Running the Pipeline

1. Update the `path_to_note` variable in `src/azure_scanner/main.py` with the path to your target handwritten clinical note or prescription image.
2. Run the pipeline as a module:
```bash
python -m src.azure_scanner.main
```

## Code Quality & Linting

This project uses `Black` for code formatting and `Ruff` for linting and code checks to maintain high code standards.

* Format code with `Black`: `python -m black .`
* Format code with `Ruff`: `ruff format .`
* Run linter checks and auto-fixes: `ruff check . --fix`
