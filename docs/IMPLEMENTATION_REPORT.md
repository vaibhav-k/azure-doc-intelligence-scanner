# Implementation Report: P0/P1 Fixes (28 October 2026)

**Scope.** This report covers changes made against the project's files as stored in the *Medcan OCR*
project. No git checkout, CI system or linked computer was available. To get a reviewable history,
I rebuilt the 18 project files into a fresh git repository and committed them unchanged as a
baseline (`4dbc392`). Every change after that is a separate commit. Line endings were normalized
from CRLF to LF.

The sample clinical image was **not** added to the repository and was not opened or processed during
this work. All tests use synthetic images generated in code.

**What was not done:**

- No call was made to Azure Document Intelligence, Azure OpenAI or Accuro, because no credentials or
  vendor access were available.
- The CI workflow was written but not executed on GitHub.
- Windows was not tested.

---

## A. Verified baseline

### Baseline commands and results

I ran these in a fresh virtual environment on Python 3.13.18. The unpinned `requirements.txt` resolved
to openai 3.40.0, pydantic 2.14.1, numpy 2.5.4, OpenCV 5.0.0 and azure-ai-documentintelligence 1.0.2.

| Command | Result |
|---|---|
| `pip install -r requirements.txt && pip install -e . && pytest` (README path) | **8 passed, 2 failed.** `test_get_azure_config_success` failed with `ValueError: too many values to unpack (expected 2)`. `test_analyze_document_success` failed with `TypeError: ... takes 1 positional argument but 3 were given`. |
| `pytest` without `pip install -e .` | **3 collection errors** (`ModuleNotFoundError: azure_scanner`) in `test_extraction`, `test_service` and `test_utils` |
| `pip install .` (non-editable), then import from outside the repo | `azure_scanner.llm_parser` fails with `ModuleNotFoundError: azure.identity`. `azure_scanner.preprocessing` fails with `ModuleNotFoundError: cv2`. |
| `azure_scanner.models is src.azure_scanner.models` | `False`: two separate module instances exist |
| Original `deskew_image` on **OpenCV 4.9.0** (the floor allowed by `requirements.txt`) | Straight page rotated **-90°**. A -3° skew was "corrected" by **-87°**. A dark-border page was rotated **-90°**. All of these are inside the original canvas, so content is cropped. |

### Status of the 12 reported findings

| # | Finding | Status |
|---|---|---|
| 1 | Two tests fail because of signature mismatches | **Verified**, as shown above |
| 2 | Mixed `azure_scanner` / `src.azure_scanner` imports | **Verified**: duplicate module objects |
| 3 | Deskew rotates or crops pages | **Verified**, and worse than the audit reported: on OpenCV 4.9 even straight pages turn 90° |
| 4 | Missing confidence treated as perfect; line-only pages escape | **Verified** in code (`getattr(word, "confidence", 1.0)`; the line branch never flags). A mutation test confirms the new tests catch a reintroduction. |
| 5 | Models cannot express uncertainty | **Verified**: every field is `Optional[str]` |
| 6 | Schema examples (fictional names and drugs) reach the LLM | **Verified**: `examples=` is present and is emitted into the strict JSON schema |
| 7 | Console prints OCR text, flagged tokens, extracted JSON and file paths | **Verified** in `main.py` and `service.py` |
| 8 | `temperature=1`, `sys.exit` in the library, no refusal or truncation handling | **Verified** |
| 9 | Package metadata omits runtime dependencies | **Verified** (the `pip install .` failures above) |
| 10 | No Accuro integration | **Verified**: no code existed |
| 11 | `pipeline.py` sketch unverified | **Verified, and the sketch was wrong.** `azure-core`'s `LROPoller.result(timeout)` *returns silently* when the timeout expires, so the sketch would have passed a timed-out operation downstream. The implementation checks `poller.done()`. |
| 12 | `gpt-chat-latest` compatibility with Structured Outputs and its parameters | **Still unverified**: no live access. See section F. |

### Additional defects found during implementation

| Defect | Evidence | Fix |
|---|---|---|
| **openai 1.106 logs the full request body (OCR text, i.e. PHI) at DEBUG** through `openai._base_client` | The privacy tests failed only on the minimum-version run. The leaked string appeared in a `Request options:` DEBUG record. | Added a logger-level guard installed on import (`install_sdk_log_guards`) and a regression test |
| openai below 1.106 does not accept a callable `api_key`, which the v1 API needs for Entra ID tokens | Inspected the SDK source of the 1.92, 1.105 and 1.106 wheels | Raised the floor to `openai>=1.106.0` |
| openai 3.x moved its HTTP stack to `httpx2` | Checked package metadata | The tests resolve the transport module from the installed SDK |
| Review approval could skip document-level flags, such as low-confidence OCR text that maps to no extracted field (a possibly omitted medication line) | Code review | Approval now requires explicit acknowledgement of every reason code |
| Priority escalation used an allow-list that missed fail-closed reasons | Code review | Any reason now escalates unless it is explicitly configured as routine |
| My own process error: `build/` artifacts were committed in an early commit | Found while auditing the wheel contents | Branch history rewritten. Every commit was scanned clean afterwards. |

---

## B. Implemented fixes

| File | Change | Rationale | Expected benefit |
|---|---|---|---|
| `pyproject.toml` | Complete runtime dependencies; `requires-python>=3.10`; src-layout discovery; `[dev]` extras; console script; ruff, mypy and pytest configuration | Finding 9 and the Python version conflict | `pip install .` works. One source of truth for dependencies. |
| `requirements.txt` | Now `-e .[dev]` | The old instructions keep working without a second dependency list | No drift |
| `requirements-lock.txt` (new) | `uv pip compile --universal --generate-hashes`, Python ≥3.10 | Reproducible, hash-checked installs | `pip-audit`: no known vulnerabilities |
| `src/azure_scanner/__init__.py` (new) | Package marker and `__version__` | Explicit package | Single namespace |
| All modules and tests | `src.azure_scanner.*` replaced by `azure_scanner.*` | Finding 2 | Mocks and `isinstance` checks work |
| `preprocessing.py` | Magic-byte format sniffing; size, dimension and blank checks (faint ink is not blank); default **`minimal` mode forwards the original bytes**; opt-in `enhanced` mode. The new deskew uses a background-normalized ink mask, removes large edge-touching dark regions, and searches projection profiles. It applies only when the angle is between 0.5° and 10° and beats 0° by at least 10%, and it rotates on an **expanded canvas**. Illumination flattening and light non-local-means denoising replace bilateral filtering and CLAHE. `deskew_image` and `preprocess_image` keep their names. | Finding 3; avoid damaging decimals and faint strokes | No 90° rotations and no cropping on OpenCV 4.9 or 5.0. Uncertain estimates are refused. |
| `confidence.py` | Fail-closed: `missing_confidence`, `invalid_confidence` (NaN, out of range, non-numeric, bool), `unscored_page` (lines only), `no_text`, `no_pages`. Word spans and polygons are kept. OCR text is excluded from `repr`. The threshold is validated. `words_in_span()` supports per-field provenance. **This is a breaking change:** the function returns `OcrConfidenceReport` instead of a tuple. | Finding 4 | No invented confidence values. Uncertainty reaches the reviewer. |
| `models.py` | `ExtractedField{verbatim, normalized, status ∈ extracted, ambiguous, illegible, not_present, alternatives}` with client-side validators. There is no "inferred" status. `ClinicalDocumentExtraction` adds route, quantity, refills and allergies. Review-layer `FieldAssessment` (OCR confidence, `found_in_source`) and `HumanConfirmation` models are separate from the LLM schema. Legacy models are kept but **examples are removed**. | Findings 5 and 6 | Uncertainty is preserved. The LLM cannot claim confidence or human confirmation. |
| `errors.py` (new) | `PipelineError` carries `code`, `stage` and `retryable`; `str(exc)` is the code only | PHI-safe errors | Actionable, safe failures |
| `logging_utils.py` (new) | Allow-listed `log_event`, a JSON formatter that renders only the exception class name, random correlation IDs, and SDK payload log guards | Finding 7 and the openai DEBUG leak | Logs carry metadata only |
| `config.py` | `pydantic-settings` `Settings`. Same environment variable names. Keys are optional (Entra ID otherwise); keys are rejected in prod; `https` is enforced; cross-field validation. No `load_dotenv` at import and no `sys.exit`. Errors list variable names only. | Finding 8 and production authentication | Testable, safe configuration |
| `service.py` | Injected client and a `build_di_client` factory (Entra ID or key, bounded `retry_total`). Works on bytes. Explicit `poller.done()` timeout check. SDK errors map to `OcrError` codes. Markdown is requested only for `prebuilt-layout`. No paths are logged. | Finding 7 and the silent poller timeout | Reliable, testable OCR stage |
| `llm_parser.py` | `LlmExtractor` with an injected client. `build_openai_client`: v1 GA by default, legacy `AzureOpenAI` when `AZURE_OPENAI_API_VERSION` is set. Uses the SDK's bounded selective retries. 13 typed errors, covering refusal, truncation, content filter, schema failure, auth, rate limit, timeout, transient, bad request (parameter name only) and not-found. Delimited untrusted-data prompt. Low-confidence span markers. Token usage is recorded. `temperature` is sent only if configured. **No unconstrained fallback.** `parse_markdown_with_gpt` was removed (breaking). | Findings 8 and 12 | No process exits. Every failure is distinguishable. |
| `review.py` (new) | Review states `pending`, `needs_review`, `approved`, `rejected`, `failed` with enforced transitions. There is **no auto-approval**. Per-field evidence: ambiguous or illegible values, OCR low or unknown confidence, `not_found_in_source` (possible hallucination), `normalization_changed_numbers`, error-prone decimals, uncertain identity. Document checks: missing core medication elements, contradictions, dose without drug, uncertain allergies. Priority is fail-safe. Approval needs a named reviewer, confirmations for every flagged and medication or allergy field, and acknowledgement of every reason code. Rejection takes a code, not free text. Events are recorded, and logs contain no identifiers. | Clinician-review requirement | Schema-valid output is never "approved" by the software |
| `pipeline.py` (new) | `DocumentPipeline.process()` runs 7 timed stages and returns `ProcessingResult`, whose validator **enforces** that completed implies `needs_review` with no error, and failed implies `failed` with an error and no extraction. Unexpected exceptions become `internal_error`. `safe_summary()` gives a PHI-free view. Everything is in memory. | Finding 11 | A failed extraction cannot look completed |
| `main.py` | `azure-scanner process ... --out-dir`. stdout gets PHI-free JSON lines. Results are written as owner-only `<correlation_id>.json` (input names, which often contain PHI, are not reused) plus a `manifest.jsonl`. Duplicates are skipped by SHA-256. Exit codes are 0, 1 and 2. | Finding 7; hardcoded path | Batch-capable, safe CLI |
| `utils.py`, `tests/test_utils.py` | **Deleted** | Dead code that printed OCR text | Less PHI surface |
| `integrations/emr.py`, `integrations/accuro.py` (new) | Vendor-neutral read-only `EmrDocumentSource` and `ResultSink` protocols; association checks (system, patient, document, version); `DisabledResultSink` default; `deliver_if_approved` requires clinician `APPROVED` status, a matching correlation ID and a `VERIFIED` capability; idempotency key. All Accuro capabilities are `UNVERIFIED`, and the vendor question list is included. | Finding 10 | No speculative endpoints. Write-back is impossible by default. |
| `.gitignore` | Ignores images, PDFs, `inputs/`, `outputs/`, manifests, `.env.*`, caches and build output | PHI in source control | Prevents committing clinical documents |
| `.env.example`, `README.md` | Rewritten: install, configuration table, offline testing, review workflow, limitations. The "production-ready" claim is removed. | Accuracy | Correct onboarding |
| `.github/workflows/ci.yml` (new) | Matrix of Python 3.10 with lowest-direct dependencies and Python 3.13 with the lock file. Runs ruff, mypy, pytest with coverage of at least 85%, and pip-audit. | Regression gate | **Not executed** (no GitHub access) |

---

## C. Test results (actual runs, final commit)

| Command | Environment | Result |
|---|---|---|
| `ruff check src tests` | dev | All checks passed |
| `ruff format --check src tests` | dev | 29 files already formatted |
| `mypy` (strict, `src/`) | dev | Success: no issues in 15 source files |
| `pytest` | **Latest**: Python 3.13.18, OpenCV 5.0.0, openai 3.40.0, pydantic 2.14.1 (from `requirements-lock.txt`) | **254 passed**, 0 failed |
| `pytest --cov=azure_scanner` | same | **254 passed, 98% line coverage** (1225 statements, 27 missed) |
| `pytest` | **Minimum**: Python 3.10.20, OpenCV 4.9.0, openai 1.106.0, pydantic 2.7.0, numpy 1.26.0, pydantic-settings 2.3.0, azure-ai-documentintelligence 1.0.0, azure-core 1.30.0 (`uv pip install --resolution lowest-direct`) | **254 passed**, 0 failed |
| `pip-audit -r requirements-lock.txt` | locked | No known vulnerabilities found |
| `python -m build --wheel`, install the wheel in a clean venv, import and run the `azure-scanner` CLI from outside the repo | Python 3.13 | Imports succeed. The CLI exits with code 2 and names the missing `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`. |

**Mutation checks.** I reintroduced 8 defects, one at a time, and every one was caught by the suite:

- missing confidence treated as 1.0
- line-only pages passing silently
- undetected poller timeout
- deskew bound removed
- refusal ignored
- approval without confirmations
- log allow-list bypassed
- EMR association check removed

**Not run or not verifiable here:**

- live Azure Document Intelligence and Azure OpenAI calls;
- Accuro;
- the GitHub Actions workflow;
- Windows;
- Python 3.11 and 3.12 individually (only the 3.10 floor and 3.13 were run; the lock file is universal);
- clinical accuracy on real handwriting.

Synthetic-image tests demonstrate **geometric safety only**, not OCR accuracy.

---

## D. Security and clinical safety

**Controls now in place:**

- No auto-approval.
- Mandatory confirmation of medication and allergy fields.
- Explicit acknowledgement of every flag.
- Detection of values not found in the source text.
- Number-changing normalization is flagged.
- Error-prone decimals are flagged.
- Fail-closed OCR confidence.
- Allow-listed logs that render no exception messages.
- SDK payload log guard.
- Owner-only result files.
- No temporary files.
- Managed identity by default, with keys refused in prod.
- `https` enforced.
- Prompt-injection envelope.
- No unconstrained parsing fallback.

**Remaining risks:**

1. **Omission.** The model may skip a medication entirely. Field-level checks cannot see what was never
   extracted. Mitigation today: document-level OCR flags must be acknowledged, and the reviewer must
   compare against the source image. A dedicated "unmapped OCR text" metric would be a useful next
   step.
2. **`found_in_source` is a substring check.** OCR errors that the model reproduces verbatim will pass
   it, which is why OCR confidence is also checked.
3. **Thresholds are uncalibrated.** The 0.85 OCR threshold and the 10° deskew bound are defaults. They
   need a de-identified evaluation set.
4. **Result files contain PHI.** Retention, encryption at rest and access control for `--out-dir` are
   organizational responsibilities. The CLI does not delete files.
5. **Process-level logging.** The guard covers the `openai` loggers. Other libraries added later must be
   assessed.
6. **Not established:** compliance with PHIPA or other provincial law, PIPEDA, or HIPAA; medical-device
   classification; Azure data residency and Azure OpenAI data-handling terms. These need legal,
   organizational and vendor verification.

---

## E. Accuro integration status

| Item | Status |
|---|---|
| `EmrDocumentSource` / `ResultSink` protocols, association checks, idempotency key, `DisabledResultSink`, `deliver_if_approved` | **Implemented and tested** with in-memory fakes (18 tests) |
| Accuro authentication, document discovery (polling, webhook or feed), content download, metadata, rate limits, write-back | **Unknown / unverified.** `ACCURO_CAPABILITIES` marks every capability `UNVERIFIED`. |
| Any Accuro endpoint, SDK or credential in code | **None**, deliberately |
| Recommended first phase | Read-only retrieval or controlled manual export, clinician review in this application, and manual entry of approved data into Accuro |

Vendor questions are listed in `integrations/accuro.py` (`VENDOR_VERIFICATION_REQUIRED`).

---

## F. Azure compatibility status

| Item | Status |
|---|---|
| `gpt-chat-latest` supports Structured Outputs (`json_schema`, strict) | **Unverified** (no deployment access). Microsoft's published supported-model list does not name it. |
| `temperature` or other sampling parameters | **Unverified.** Not sent unless `LLM_TEMPERATURE` is set. A rejection surfaces as `LlmRequestError(detail="temperature")` without a retry. |
| Schema acceptance | Verified **offline only**: strict schema generated by the SDK; every property required; `additionalProperties: false`; none of Microsoft's documented unsupported keywords; property count ≤100; nesting ≤5 |
| v1 API path and legacy `api-version` path | URL construction verified offline through the SDK mock transport |
| Entra ID token scopes (`https://ai.azure.com/.default` for v1, `https://cognitiveservices.azure.com/.default` for legacy) | Taken from Microsoft documentation. Overridable with `AZURE_OPENAI_TOKEN_SCOPE`. **Unverified live.** |
| Document Intelligence span offsets with Markdown output | Assumed to index `result.content`. **Unverified live.** |

**Live verification step.** Run one call against the actual deployment with a **synthetic** document
and `ENABLE_LLM=true`, then record the API version, the model version behind `gpt-chat-latest`, and the
result.

---

## G. Remaining work, prioritized

1. **Live smoke test** with a synthetic document against both Azure services. Confirm Structured
   Outputs, the temperature setting, token scopes and span offsets. Pin a dated model version rather
   than "latest".
2. **De-identified evaluation set and harness** measuring CER/WER, per-field medication accuracy and
   the rate at which errors are caught by flags. Use it to calibrate the OCR threshold and decide
   whether `PREPROCESS_MODE=enhanced` helps.
3. **Reviewer UI**: image region (polygon) beside each field, confirm, edit or reject, and acknowledgement
   of reasons. The backend state machine exists.
4. **Persistence and audit store** for `ReviewRecord` events (append-only), plus a retention and
   deletion job for `--out-dir`.
5. **Accuro**: send the vendor question list, obtain sandbox access, then build a read-only adapter
   against `EmrDocumentSource`.
6. **Privacy impact assessment**, Azure region and deployment-type decision, and agreements with
   Microsoft and QHR.
7. Run CI on GitHub and add Python 3.11 and 3.12 to the matrix.
8. Omission detection: flag OCR text with no counterpart in the extraction.

**Prerequisites for production:** items 1, 2, 3, 4 and 6 at minimum.

---

## H. Final change list (baseline `4dbc392` to HEAD)

**Modified:**

- `.env.example`, `.gitignore`, `README.md`, `pyproject.toml`, `requirements.txt`
- `src/azure_scanner/confidence.py`, `config.py`, `llm_parser.py`, `main.py`, `models.py`,
  `preprocessing.py`, `service.py`
- `tests/test_config.py`, `tests/test_extraction.py`, `tests/test_service.py`

**Created:**

- `.github/workflows/ci.yml`, `requirements-lock.txt`, `docs/IMPLEMENTATION_REPORT.md`
- `src/azure_scanner/__init__.py`, `errors.py`, `logging_utils.py`, `pipeline.py`, `review.py`
- `src/azure_scanner/integrations/__init__.py`, `integrations/emr.py`, `integrations/accuro.py`
- `tests/conftest.py`, `tests/factories.py`, `tests/test_confidence.py`, `tests/test_emr_integration.py`,
  `tests/test_llm_parser.py`, `tests/test_logging.py`, `tests/test_main.py`, `tests/test_pipeline.py`,
  `tests/test_preprocessing.py`, `tests/test_review.py`

**Deleted:**

- `src/azure_scanner/utils.py`, `tests/test_utils.py`

**Unchanged:** `tests/__init__.py`

**Breaking API changes:**

- `get_azure_config()` is replaced by `load_settings()`.
- `parse_markdown_with_gpt()` is replaced by `LlmExtractor`.
- `evaluate_ocr_confidence()` returns `OcrConfidenceReport` instead of a tuple.
- `DocumentScannerService.__init__` takes a client instead of an endpoint and key.
- `print_extracted_text` is removed.
- The CLI now requires `process <files> --out-dir`.
