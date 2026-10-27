# Azure Document Intelligence Note Scanner

A lightweight, robust Python script to extract handwritten and printed text from medical notes, scanned prescriptions, or documents using the modern Azure Document Intelligence SDK (prebuilt-read model).

## Prerequisites

* Python 3.8+

* An active Azure Document Intelligence resource in your Azure portal.

## Setup Instructions

1. Clone or download this repository (`git clone https://github.com/vaibhav-k/azure-doc-intelligence-scanner.git`).

2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install dependencies: `pip install -r requirements.txt`

4. Configure your credentials:
   - Copy the template file: `cp .env.example .env`

   - Open .env and fill in your Azure Document Intelligence ENDPOINT and KEY.

## Running the Script

1. Update the `PATH_TO_NOTE` variable in `main.py` with the path to your target image or PDF.

2. Run the script: `python main.py`

## Code Quality & Linting

This project uses `Black` for code formatting and `Ruff` for linting and code checks to maintain high code standards.

- Format code with `Black`: `python -m black .`

- Format with `Ruff`: `ruff format .`

- Run linter checks and auto-fixes: `ruff check . --fix`
