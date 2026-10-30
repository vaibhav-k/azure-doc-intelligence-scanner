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
| `preprocessing.py` | Format sniffing, size and blank-page checks, local page counting (PDF via PDFium, TIFF via its page directory). Default `minimal` mode forwards the original image. Opt-in `enhanced` mode applies bounded deskew (±10°, never crops, refuses uncertain estimates), illumination flattening and light denoise. |
| `service.py` | Document Intelligence adapter: injected client, bounded SDK retries, explicit timeout detection, coded errors. |
| `confidence.py` | Word-level OCR confidence. Missing or invalid scores are *unknown*, never 1.0. Line-only pages are flagged. |
| `models.py` | Model-facing schema (`verbatim` / `normalized` / `status ∈ {extracted, ambiguous, illegible, not_present}` / `alternatives`) and review-layer models (OCR confidence, source provenance, human confirmations). |
| `llm_parser.py` | Azure OpenAI client (v1 GA API by default, or legacy `api-version`), typed errors, refusal, truncation and schema-failure handling. There is no unconstrained fallback. |
| `review.py` | Deterministic routing and the review state machine (`pending → needs_review → approved/rejected`, or `failed`). |
| `pipeline.py` | Orchestration. `ProcessingResult` enforces that a failed run can never look completed. |
| `config.py` | Typed, validated settings from environment variables / `.env`. Errors name variables, never values; mistyped setting names are reported. |
| `credentials.py` | One Microsoft Entra ID credential per process, shared by both Azure clients: managed identity in `prod`, `DefaultAzureCredential` in `dev`/`test`. |
| `errors.py` | PHI-free exception hierarchy: every error has a stable `code`, a `stage` and a `retryable` flag. |
| `logging_utils.py` | Allow-listed JSON logs with correlation IDs. No document text, file paths or exception messages. |
| `integrations/` | Vendor-neutral, read-only EMR contract. Accuro capabilities are recorded as **unverified**. |
| `main.py` | CLI: input expansion (files, folders, wildcards), private result files, PHI-free summaries, resource cleanup. |

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

Copy `.env.example` to `.env` and edit it. Settings come from environment variables first, then from the
`.env` file. Variable names are case-insensitive, and names from earlier versions still work.

| Variable | Default | Notes |
|---|---|---|
| `ENVIRONMENT` | `dev` | `dev`, `test` or `prod`. `prod` refuses API keys and authenticates **only** with a managed identity (`ManagedIdentityCredential`), as Microsoft recommends for deployed apps. `dev`/`test` use `DefaultAzureCredential`, so `az login` works locally. |
| `AZURE_CLIENT_ID` | unset | Client ID of a user-assigned managed identity. Unset uses the system-assigned identity. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. Azure, OpenAI and HTTP library loggers stay at `WARNING` regardless. |
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` | — (required) | Must start with `https://`. |
| `AZURE_DOCUMENT_INTELLIGENCE_KEY` | unset | Unset means Microsoft Entra ID (see `ENVIRONMENT`). The identity needs the **Cognitive Services User** role on the resource, and the endpoint must be the resource's custom subdomain (`https://<name>.cognitiveservices.azure.com/`), not a regional endpoint. Once Entra ID works, disable key authentication on the resource. |
| `AZURE_DOCUMENT_INTELLIGENCE_MODEL` | `prebuilt-layout` | Markdown output is requested only for `prebuilt-layout`. |
| `OCR_TIMEOUT_SECONDS` | `120` | 0 < x ≤ 900. A slower operation fails with `ocr_timeout`. |
| `OCR_MAX_RETRIES` | `3` | 0–6. Bounded azure-core retries, which honour `Retry-After`. |
| `OCR_CONFIDENCE_THRESHOLD` | `0.85` | 0 < x ≤ 1. Words below this are flagged. **Calibrate it on a de-identified evaluation set.** |
| `PREPROCESS_MODE` | `minimal` | `minimal` forwards the original image. `enhanced` is opt-in. Enable it only if measurements show a benefit. |
| `MAX_INPUT_BYTES` | `52428800` (50 MiB) | Up to 500 MiB. Larger inputs fail with `input_too_large`. |
| `ENABLE_LLM` | `false` | `true` requires `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_DEPLOYMENT_NAME`. |
| `AZURE_OPENAI_ENDPOINT` | unset | Must start with `https://`. |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | unset | The deployment name, not the model name. |
| `AZURE_OPENAI_API_KEY` | unset | Unset means Entra ID; the identity needs the **Cognitive Services OpenAI User** role. Refused when `ENVIRONMENT=prod`. |
| `AZURE_OPENAI_API_VERSION` | unset | Unset uses the v1 GA API (`<endpoint>/openai/v1/`). Set it (e.g. `2024-10-01-preview`) to use the legacy `AzureOpenAI` client. |
| `AZURE_OPENAI_TOKEN_SCOPE` | unset | Overrides the Entra ID scope (`https://ai.azure.com/.default` for v1, `https://cognitiveservices.azure.com/.default` for legacy). |
| `LLM_TEMPERATURE` | unset | 0–2. When unset, it is **not sent**. Set it only after confirming that your deployment supports it. |
| `LLM_TIMEOUT_SECONDS` | `60` | 0 < x ≤ 600. |
| `LLM_MAX_RETRIES` | `2` | 0–5. The SDK retries only connection errors, timeouts, 408, 409, 429 and 5xx. |
| `LLM_MAX_INPUT_CHARS` | `60000` | OCR text longer than this fails with `document_text_exceeds_limit` rather than being truncated. |
| `LLM_MAX_OUTPUT_TOKENS` | unset | Caps generated tokens per document (sent as `max_completion_tokens`) to bound cost and latency. Unset = not sent (the configuration verified live). Reasoning models count reasoning tokens against it, so leave headroom; hitting it fails with `output_truncated`. |

Configuration errors name the variable but never echo its value. Library code never calls `sys.exit`.
Variables under this app's prefixes (`AZURE_DOCUMENT_INTELLIGENCE_`, `AZURE_OPENAI_`, `OCR_`, `LLM_`,
`PREPROCESS_`) that no setting reads, usually typos such as `OCR_CONFIDENCE_TRESHOLD`, are reported on
stderr as `configuration warning: ...` (names only) and otherwise ignored.

## Running

### 1. One-time setup

From the repository root, with the virtual environment active (see [Installation](#installation)):

```powershell
pip install -e ".[dev]"            # installs the package and the `azure-scanner` command
Copy-Item .env.example .env        # macOS/Linux: cp .env.example .env
```

Edit `.env` and set at least `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`. Either set
`AZURE_DOCUMENT_INTELLIGENCE_KEY` (dev only) or sign in for Entra ID with `az login`. To also run LLM
extraction, set `ENABLE_LLM=true`, `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_DEPLOYMENT_NAME`.

### 2. Process documents

The CLI has one sub-command, `process`. It needs one or more inputs and `--out-dir`. An input can be a
file, a folder (its files, not subfolders) or a wildcard such as `.\inputs\*.jpg`; wildcards are expanded
by the tool because PowerShell does not expand them. Supported formats are JPEG, PNG, BMP, TIFF, PDF and
HEIC/HEIF (iPhone photos); the format is detected from the file content, not the extension. HEIF files are
sent to Document Intelligence unchanged: they cannot be preprocessed or page-counted locally.

```powershell
# OCR + confidence assessment only (no Azure OpenAI call)
azure-scanner process ".\inputs\scan1.jpg" --out-dir .\results --no-llm

# Full pipeline (OCR + structured extraction), several files at once
azure-scanner process .\inputs\scan1.jpg .\inputs\scan2.pdf --out-dir .\results

# Every file in a folder, or a wildcard
azure-scanner process .\inputs --out-dir .\results
azure-scanner process ".\inputs\*.jpg" --out-dir .\results

# Use a different settings file (it must exist)
azure-scanner process .\inputs\scan1.jpg --out-dir .\results --env-file .\.env.test

# Keep the operational logs in a file and only the summaries on screen
azure-scanner process .\inputs\scan1.jpg --out-dir .\results 2> .\scanner.log

# Equivalent without the console script
python -m azure_scanner.main process .\inputs\scan1.jpg --out-dir .\results
```

Quote any path that contains spaces or brackets. Run `azure-scanner process --help` for all options.

> **Common mistakes**
>
> - `python -m src.azure_scanner.main ...`: do **not** include `src.`. The package is `azure_scanner`;
>   importing it as `src.azure_scanner` loads a second copy of every module.
> - Running without `process` gives `error: the following arguments are required: command`.
> - `azure-scanner` not found: the virtual environment is not active, or `pip install -e .` was not run.
> - `configuration error: ...` (exit code 2): a required variable is missing or invalid. The message
>   names the variable but never prints its value.

### 3. Read the output

A run writes to three places:

| Where | What | Contains PHI? |
|---|---|---|
| **stderr** | JSON log lines (`input_prepared`, `ocr_completed`, `review_status_changed`, `document_processed`, ...) with correlation ID, timings and counts | No (allow-listed fields only) |
| **stdout** | One JSON summary line per input (shown below) | No |
| **`--out-dir`** | `<correlation_id>.json` (the full result, including `ocr_content`: the complete OCR text) and `manifest.jsonl` (maps input paths to correlation IDs) | **Yes** |

Example stdout line from an OCR-only run:

```json
{"index": 0, "correlation_id": "466376e4...", "processing_status": "completed",
 "review_status": "needs_review", "priority": "priority",
 "reason_codes": ["llm_disabled_manual_transcription_required", "ocr_low_confidence"],
 "failed_stage": null, "error_code": null, "error_detail": null, "retryable": false, "page_count": 1,
 "medication_count": null,
 "timings_ms": {"input_validation": 52, "ocr": 10014, "confidence_assessment": 5, "review_routing": 0}}
```

- `processing_status` is `completed` or `failed`. `completed` always means `needs_review`, never approved.
- `reason_codes` explain why the document needs review (see [Review reason codes](#review-reason-codes)).
- On failure, `failed_stage` and `error_code` say what went wrong (see [Error codes](#error-codes)), and
  `retryable: true` means running the same input again may succeed. `error_detail` adds a PHI-free hint
  where one exists: an HTTP status, a rejected parameter name, or `returned/expected` page counts.
- `timings_ms` includes the stage that failed, so slow failures and timeouts are visible.
- `medication_count` is `null` when the LLM did not run.
- `ocr_content` in the result file holds the full OCR text whenever OCR succeeded, including OCR-only
  (`--no-llm`) runs and runs where the LLM step failed. Use it for manual transcription and review. It
  is never written to stdout or the logs.
- Identical inputs in the same run are processed once; repeats are reported as `skipped_duplicate`.

Result files are created with mode `0600` and the directory with `0700`. **On Windows these modes are not
enforced**: the files inherit the folder's access-control list, so keep `--out-dir` in a location only you
(or the intended reviewers) can read. The tool does not delete result files; apply your retention policy.

Exit codes:

- `0`: all inputs completed (they still need review).
- `1`: one or more inputs failed.
- `2`: usage or configuration error.

## Clinician review

- Completed documents always enter `needs_review`, with priority `routine` or `priority`. There is no
  auto-approve path.
- Review reasons are PHI-free codes (listed below). Every reason makes the document `priority`
  unless it is configured as routine in `ReviewPolicy.routine_reasons` (empty by default).
- `review.approve()` requires:
  - a named reviewer;
  - a `HumanConfirmation` for every flagged field and every medication or allergy field;
  - explicit acknowledgement of every document-level reason code.
- A schema-valid extraction is never approved for that reason alone. Documents processed without the LLM
  cannot be approved; they require manual transcription, using `ocr_content` and the source image.

### Review reason codes

| Code | Level | Meaning |
|---|---|---|
| `llm_disabled_manual_transcription_required` | document | Run without the LLM. Nothing can be approved; a clinician must transcribe manually. |
| `extraction_unavailable` | document | The LLM was enabled but produced no extraction. Manual transcription required. |
| `ocr_low_confidence` | document / field | At least one word is below `OCR_CONFIDENCE_THRESHOLD`. |
| `ocr_missing_confidence`, `ocr_invalid_confidence` | document | The OCR engine returned no or an unusable confidence score for some words. |
| `ocr_unscored_page` | document | A page has lines but no word-level scores. |
| `ocr_no_pages`, `ocr_no_text` | document | OCR returned no pages, or no text. |
| `ocr_confidence_unknown` | field | Some words behind this field have no confidence score. |
| `model_reported_warnings` | document | The model reported reading problems of its own. |
| `allergy_uncertain` | document | An allergy entry is ambiguous, illegible or has no reading. |
| `extracted_despite_low_ocr_confidence` | field | The model marked the value as clearly written (`extracted`), but OCR scored the text behind it below the threshold. A confident value on weak evidence: **check these first**. Always `priority`; it cannot be configured as routine. |
| `ambiguous`, `illegible` | field | The model marked the value as uncertain. |
| `not_found_in_source` | field | The model's verbatim text does not appear in the OCR text (possible hallucination). |
| `normalization_changed_numbers` | field | The normalized value has different numbers from the verbatim reading. |
| `error_prone_decimal` | field | A trailing zero (`5.0 mg`) or naked decimal (`.5 mg`) in strength, dose or quantity. |
| `identity_uncertain` | field | Patient name or date of birth/age is uncertain. |
| `medication_drug_name_missing`, `medication_strength_missing`, `medication_frequency_missing` | medication | A required medication element is not present. |
| `dose_without_drug_name` | medication | A strength or dose was read with no drug name. |
| `contradictory_medication_entries` | medication | The same drug appears with different strengths. |

### Error codes

`error_code` in the stdout summary is one of these PHI-free codes. Codes marked ⟳ are reported with
`retryable: true` (some only for specific HTTP statuses).

| Stage | Codes |
|---|---|
| Input / preprocessing | `input_unreadable`, `empty_input`, `unsupported_format`, `input_too_large`, `corrupt_or_undecodable_image`, `image_dimensions_too_large` (over 120 megapixels; checked from the file header before decoding), `image_too_small` (shorter side < 50 px), `blank_or_unreadable_page`, `corrupt_or_undecodable_pdf`, `pdf_password_protected` (remove the password first), `too_many_pages` (over 2,000, Document Intelligence's limit), `encode_failed` |
| OCR (Document Intelligence) | `ocr_auth_failed`, `ocr_timeout` ⟳, `ocr_connection_error` ⟳, `ocr_http_<status>` (⟳ for 408, 429, 5xx), `ocr_empty_result`, `ocr_no_text_detected`, `ocr_page_count_mismatch` (the service returned fewer pages than the input has; see below) |
| LLM (Azure OpenAI) | `llm_endpoint_not_configured`, `llm_deployment_not_configured`, `llm_deployment_not_found`, `llm_auth_failed`, `llm_rate_limited` ⟳, `llm_timeout` ⟳, `llm_connection_error` ⟳, `llm_server_error` ⟳, `llm_bad_request` (a 4xx, e.g. an unsupported parameter such as `temperature`), `llm_http_error`, `empty_document_text`, `document_text_exceeds_limit`, `prompt_content_filtered`, `content_filtered`, `model_refused`, `output_truncated`, `unparsed_response`, `schema_validation_failed` |
| Output | `output_write_failed` (the result could not be saved, e.g. disk full; the batch continues) |
| Any | `internal_error` (an unexpected exception; only its class name is logged) |

**Page coverage check.** The tool counts the input's pages locally and fails the document with
`ocr_page_count_mismatch` if Document Intelligence returns fewer. The usual cause is the **free (F0) tier,
which analyses only the first 2 pages** of a PDF or TIFF; use the paid (S0) tier for multi-page documents.
Partial OCR of a clinical document is never reported as completed.

## Testing (offline, no credentials)

```bash
pytest                                   # whole suite; never calls Azure or any EMR
pytest --cov=azure_scanner               # coverage
ruff check src tests                     # lint
ruff format --check src tests            # formatting
mypy                                     # strict type check of src/
pyright                                  # src/ and tests/, same checks Pylance runs in VS Code
```

- An autouse fixture clears `AZURE_*` and related variables and runs each test in an empty temporary
  directory, so a developer's `.env` cannot affect results.
- All images are generated synthetically in `tests/conftest.py`.
- LLM tests drive the real OpenAI SDK through an in-memory HTTP transport.
- Synthetic-image tests check geometric safety only. **They do not measure clinical OCR accuracy.**

CI (`.github/workflows/ci.yml`) runs ruff, mypy, pyright, pytest with at least 85% coverage, and `pip-audit`, on
Python 3.10 with the lowest allowed dependency versions and on Python 3.13 with `requirements-lock.txt`.

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
| `gpt-chat-latest` deployment: Structured Outputs support | **Verified live on 2026-10-28** (one document, pipeline 0.2.0, prompt `2026-10-28.2`, schema `2.0.0`): the strict schema was accepted, `finish_reason` was `stop` and the response parsed with no schema errors. Configuration used: legacy `AzureOpenAI` client with `api-version` `2024-12-01-preview`; API-key authentication for both Azure OpenAI and Document Intelligence (`ENVIRONMENT=dev`); `LLM_TEMPERATURE` unset, so no temperature was sent. Still to record: the model version behind `gpt-chat-latest` (prompt `2026-10-28.2`, schema `2.0.0`). The v1 GA API surface and Entra ID authentication have **not** been exercised live. "latest" can change underneath you; re-run the smoke test after any model update. |
| `temperature` and other sampling parameters | **Unverified.** The live run did not exercise an explicit value. Leave `LLM_TEMPERATURE` unset until one is tested. |
| Document Intelligence live call (`prebuilt-layout`, 1-page JPEG) | **Verified live on 2026-10-28**: OCR completed in about 7 s with word-level confidence scores. |
| Markdown output for models other than `prebuilt-layout` | Not requested, because it is undocumented for `prebuilt-read`. |
| Document Intelligence span offsets with Markdown output | Assumed to index `result.content`. Verify with a live call on a synthetic document. |
| Managed identity (`ENVIRONMENT=prod`) and the shared credential | Construction tested offline; **not exercised against Azure**. |
| `LLM_MAX_OUTPUT_TOKENS` (`max_completion_tokens`) | Request shape tested offline; **not verified** against your deployment and `api-version`. |
| HEIC/HEIF input | Accepted per Microsoft's documented formats; **not verified live**. |
| OCR timeouts | A timed-out analysis keeps running, and billing, in Azure; it cannot be cancelled, so retrying `ocr_timeout` pays for the document again. |
| Accuro EMR: authentication, discovery, download, rate limits, write-back | **Unknown.** See `src/azure_scanner/integrations/accuro.py` for the vendor questions. Write-back is refused by default. |
| Clinical accuracy on real handwriting | Not measured. Requires the evaluation set above. |
| Data residency, Azure OpenAI data-handling terms, privacy impact assessment, medical-device classification | Organizational, legal and vendor verification required. |

## Remaining work before production

1. **Finish the live smoke test.** Done on 2026-10-28: both Azure services respond, and Structured Outputs
   is accepted. Still open: record the API version, auth method and model version (see the table above);
   test an explicit `temperature`; confirm Markdown span offsets; test Entra ID token scopes; pin a dated
   model version rather than "latest". Use synthetic documents for these runs.
2. **De-identified evaluation set and harness** (see [Clinical evaluation](#clinical-evaluation-not-yet-built)).
3. **Reviewer UI**: image region (polygon) beside each field; confirm, edit or reject; acknowledge reasons.
   The backend state machine exists.
4. **Persistence and audit store** for `ReviewRecord` events (append-only), plus a retention and deletion
   job for `--out-dir`.
5. **Accuro**: send the vendor question list, obtain sandbox access, then build a read-only adapter against
   `EmrDocumentSource`.
6. **Privacy impact assessment**, Azure region and deployment-type decision, and agreements with Microsoft
   and QHR.
7. Run CI on GitHub and add Python 3.11 and 3.12 to the matrix.
8. Omission detection: flag OCR text with no counterpart in the extraction.

Items 1, 2, 3, 4 and 6 are the minimum prerequisites for production use.
