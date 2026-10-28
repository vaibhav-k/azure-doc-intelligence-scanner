# Azure Clinical Document Scanner

Extracts handwritten clinical notes and prescriptions using **Azure Document Intelligence** (OCR),
optional **OpenCV** preprocessing, and **Azure OpenAI Structured Outputs** validated by **Pydantic v2**.
Every result is routed to **clinician review**. The software never approves, prescribes or writes to a
medical record on its own.

> **Status: pre-production.** The offline test suite passes, but live Azure compatibility, clinical
> accuracy on real documents, and Accuro EMR capabilities are **not yet verified** (see
> [Known limitations](#known-limitations-and-unverified-items)). Nothing here establishes compliance with
> any healthcare privacy or medical-device regulation.

## How it works

```
input bytes ─► validation ─► preprocessing ─► Document Intelligence OCR ─► OCR confidence assessment
            ─► Azure OpenAI structured extraction ─► schema + cross-field validation
            ─► review routing (needs_review, routine | priority) ─► ProcessingResult
```

| Module | Responsibility |
|---|---|
| `preprocessing.py` | Format sniffing, size and blank-page checks. Default `minimal` mode forwards the original image. Opt-in `enhanced` mode applies bounded deskew (±10°, never crops, refuses uncertain estimates), illumination flattening and light denoise. |
| `service.py` | Document Intelligence adapter: injected client, bounded SDK retries, explicit timeout detection, coded errors. |
| `confidence.py` | Word-level OCR confidence. Missing or invalid scores are *unknown*, never 1.0. Line-only pages are flagged. |
| `models.py` | Model-facing schema (`verbatim` / `normalized` / `status ∈ {extracted, ambiguous, illegible, not_present}` / `alternatives`) and review-layer models (OCR confidence, source provenance, human confirmations). |
| `llm_parser.py` | Azure OpenAI client (v1 GA API by default, or legacy `api-version`), typed errors, refusal, truncation and schema-failure handling. There is no unconstrained fallback. |
| `review.py` | Deterministic routing and the review state machine (`pending → needs_review → approved/rejected`, or `failed`). |
| `pipeline.py` | Orchestration. `ProcessingResult` enforces that a failed run can never look completed. |
| `logging_utils.py` | Allow-listed JSON logs with correlation IDs. No document text, file paths or exception messages. |
| `integrations/` | Vendor-neutral, read-only EMR contract. Accuro capabilities are recorded as **unverified**. |
| `main.py` | CLI. |

## Installation

Requires Python **3.10+**.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Reproducible install (exact, hash-checked versions that were tested):
pip install -r requirements-lock.txt
pip install --no-deps -e .

# Or a development install that resolves the ranges in pyproject.toml:
pip install -e ".[dev]"
```

`pyproject.toml` is the single source of truth for dependencies. `requirements.txt` only points at it.
The project uses **`opencv-python-headless`**. If an older `opencv-python` is installed in the same
environment, uninstall it first (`pip uninstall opencv-python`), because both provide `cv2`.

## Configuration

Copy `.env.example` to `.env` and edit it. Existing variable names from earlier versions still work.

| Variable | Default | Notes |
|---|---|---|
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` | — (required) | Must be `https://`. |
| `AZURE_DOCUMENT_INTELLIGENCE_KEY` | unset | Unset means Microsoft Entra ID (`DefaultAzureCredential`, for example a managed identity). Keys are rejected when `ENVIRONMENT=prod`. |
| `AZURE_DOCUMENT_INTELLIGENCE_MODEL` | `prebuilt-layout` | Markdown output is requested only for `prebuilt-layout`. |
| `OCR_CONFIDENCE_THRESHOLD` | `0.85` | Words below this are flagged. **Calibrate it on a de-identified evaluation set.** |
| `OCR_TIMEOUT_SECONDS`, `OCR_MAX_RETRIES` | `120`, `3` | Bounded azure-core retries, which honour `Retry-After`. |
| `PREPROCESS_MODE` | `minimal` | `enhanced` is opt-in. Enable it only if measurements show a benefit. |
| `ENABLE_LLM` | `false` | Requires `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_DEPLOYMENT_NAME`. |
| `AZURE_OPENAI_API_VERSION` | unset | Unset uses the v1 GA API (`<endpoint>/openai/v1/`). Set it to use the legacy `AzureOpenAI` client. |
| `LLM_TEMPERATURE` | unset | When unset, it is **not sent**. Set it only after confirming that your deployment supports it. |
| `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES` | `60`, `2` | The SDK retries only connection errors, timeouts, 408, 409, 429 and 5xx. |

Configuration errors name the variable but never echo its value. Library code never calls `sys.exit`.

## Running

```bash
azure-scanner process scan1.jpg scan2.pdf --out-dir ./results
# or: python -m azure_scanner.main process scan1.jpg --out-dir ./results [--no-llm]
```

**stdout** shows one PHI-free JSON line per input: index, correlation ID, status, review status, priority
and reason codes.

**`--out-dir`** contains the full results, which **contain PHI**:

- `<correlation_id>.json`, created with owner-only permissions. Input file names are not reused.
- `manifest.jsonl`, which maps input paths to correlation IDs.

The tool does not delete these files. Apply your retention policy to them.

Exit codes:

- `0`: all inputs completed (they still need review).
- `1`: one or more inputs failed.
- `2`: usage or configuration error.

## Clinician review

- Completed documents always enter `needs_review`, with priority `routine` or `priority`. There is no
  auto-approve path.
- Review reasons are PHI-free codes, for example:
  - `ambiguous`, `illegible`
  - `ocr_low_confidence`, `ocr_confidence_unknown`
  - `not_found_in_source`: the model's verbatim text does not appear in the OCR text
  - `normalization_changed_numbers`
  - `error_prone_decimal` (`.5 mg`, `5.0 mg`)
  - `contradictory_medication_entries`, `dose_without_drug_name`
  - `medication_<field>_missing`
  - `allergy_uncertain`, `identity_uncertain`
- `review.approve()` requires:
  - a named reviewer;
  - a `HumanConfirmation` for every flagged field and every medication or allergy field;
  - explicit acknowledgement of every document-level reason code.
- A schema-valid extraction is never approved for that reason alone. Documents processed without the LLM
  cannot be approved; they require manual transcription.

## Testing (offline, no credentials)

```bash
pytest                                   # whole suite; never calls Azure or any EMR
pytest --cov=azure_scanner               # coverage
ruff check src tests && mypy             # lint + type check
```

- An autouse fixture clears `AZURE_*` and related variables and runs each test in an empty temporary
  directory, so a developer's `.env` cannot affect results.
- All images are generated synthetically in `tests/conftest.py`.
- LLM tests drive the real OpenAI SDK through an in-memory HTTP transport.
- Synthetic-image tests check geometric safety only. **They do not measure clinical OCR accuracy.**

### Verifying minimum versions

```bash
uv venv -p 3.10 .venv-min && uv pip install -p .venv-min --resolution lowest-direct -e ".[dev]"
.venv-min/bin/pytest
```

### Clinical evaluation (not yet built)

Before any production use, assemble a **de-identified or synthetic** document set with ground truth, then
measure:

- character error rate (CER) and word error rate (WER);
- per-field medication accuracy;
- the rate at which errors are caught by review flags.

Use these measurements to calibrate `OCR_CONFIDENCE_THRESHOLD` and choose `PREPROCESS_MODE`.

## Data handling rules for contributors

- Never commit real clinical documents. `.gitignore` excludes images, PDFs, `inputs/` and `outputs/`.
  If a real document was ever committed, purge it from history (for example with `git filter-repo`) and
  inform your privacy officer.
- Do not add document text, file paths, patient or document identifiers, or exception messages to logs.
  Use `log_event` with allow-listed fields.
- Treat every model output as untrusted until a clinician has reviewed it.

## Known limitations and unverified items

| Item | Status |
|---|---|
| `gpt-chat-latest` deployment: Structured Outputs support, `temperature`, other parameters | **Unverified.** No live access was available. The schema is checked offline against Microsoft's documented Structured Outputs subset. |
| Markdown output for models other than `prebuilt-layout` | Not requested, because it is undocumented for `prebuilt-read`. |
| Document Intelligence span offsets with Markdown output | Assumed to index `result.content`. Verify with a live call on a synthetic document. |
| Accuro EMR: authentication, discovery, download, rate limits, write-back | **Unknown.** See `src/azure_scanner/integrations/accuro.py` for the vendor questions. Write-back is refused by default. |
| Clinical accuracy on real handwriting | Not measured. Requires the evaluation set above. |
| Data residency, Azure OpenAI data-handling terms, privacy impact assessment, medical-device classification | Organizational, legal and vendor verification required. |
