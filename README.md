# Rewiseorist

Rewiseorist is a free, local-first revision organiser for students. You upload your own
notes and documents, get a study plan, chat with a source-grounding revision assistant,
take checked quizzes, track progress and earn XP — all running on your own Windows
machine. Nothing has to leave the device: revision chat runs against a local Ollama
server, and there is no paywall (compulsory spend is AED 0).

## Approved delivery change (AC-6)

**Revision chat now uses the pretrained SmolLM2 360M model through Ollama's local HTTP
API.** This replaces the previous from-scratch language-model requirement for this
delivery (approved change AC-6).

- Model: `smollm2:360m` (Ollama tag id `297281b699fc`, ~725 MB, pulled 2026-11-18)
- Ollama 0.40.16, base URL `http://127.0.0.1:11434`, endpoint `POST /api/chat`
- The from-scratch checkpoints in `models/` are **preserved unchanged as training
  artefacts but are NOT used to answer revision questions**.

## Requirements

- Windows (developed and tested on Windows 11)
- Python 3.14 (tested with 3.14.7) — see `requirements.txt`
- [Ollama](https://ollama.com) installed and running locally (tested with 0.40.16)

## Install

```powershell
python -m venv .venv
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt
```

## Launch and stop

```powershell
# 1. Start Ollama (skip if the Ollama app is already running)
ollama serve

# 2. Pull the model once
ollama pull smollm2:360m

# 3. Start Rewiseorist
& ".venv\Scripts\python.exe" app.py
```

Open <http://127.0.0.1:5000>.

Stop with `Ctrl+C` in the terminal running the app; quit Ollama from its app (or stop
`ollama serve` with `Ctrl+C`).

## Configuration

All keys are set in `app.py` (`OLLAMA_*` config) and read by `OllamaClient.from_config`
in `ml/ollama_client.py`:

| Key | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Local Ollama server base URL |
| `OLLAMA_MODEL` | `smollm2:360m` | Model tag sent to `/api/chat` |
| `OLLAMA_NUM_CTX` | `2048` | Context window (tokens) |
| `OLLAMA_MAX_TOKENS` | `200` | Maximum answer length (`num_predict`) |
| `OLLAMA_TEMPERATURE` | `0.2` | Sampling temperature |
| `OLLAMA_READ_TIMEOUT` | `90.0` | Read timeout in seconds |

Fixed client constants (not config keys): connect timeout 3.05 s, `keep_alive=5m`,
top 3 sources, 600 characters per source.

## How revision chat works

1. `POST /api/chat` retrieves candidate passages for the topic from your uploaded
   resources (`resources.retrieve`, top 5).
2. `ml/ollama_client.py` keeps the **top 3 distinct passages, each capped at 600
   characters** (`pick_sources`).
3. `build_ollama_revision_messages` (`ml/prompts.py`) builds a short two-message
   prompt; the client calls `POST http://127.0.0.1:11434/api/chat` with
   `num_ctx=2048`, `num_predict=200`, `temperature=0.2`, `keep_alive=5m`.
4. **Grounding check:** the answer is scored by lexical overlap against the picked
   sources; citations that name no real retrieved source are stripped
   (`validate_citation`). The grounding status (`verified` / `unverified` /
   `no-source`) is stored with the message and returned in the API response.
5. The answer is saved with its sources and rendered in the Study chat with
   **source chips** (file name + reference).

Typed failures (`OllamaUnavailable`, `OllamaModelMissing`, `OllamaTimeout`,
`OllamaEmptyAnswer`, `OllamaHTTPError`) become clear HTTP 502/503/504 messages that
tell the learner exactly what to run (e.g. `ollama serve`, `ollama pull smollm2:360m`).

`GET /api/models/status` and `GET /api/health` include an `ollama` status block;
`POST /api/models/warmup` and `POST /api/models/unload` manage both Ollama and the
classifier worker.

### Measured smoke result

On 2026-11-18, a live smoke request to `smollm2:360m` returned **"OK" in 3.73 s**
(evidence: `evidence/verification/ollama-smoke.json`; Ollama version:
`evidence/verification/ollama-version.txt`).

## Performance classification (current fallback)

`models/performance_model.joblib` and `models/performance_preprocessor.json` are
**absent**. `POST /api/classify` therefore uses a transparent rule-based fallback
returning `model_version: "rule-fallback-v1"` and `fallback: true` (function
`rule_label` in `ml/train_character.py`). This is a documented remaining gap, not a
trained classifier.

Quiz marking, scheduling, rewards and calculations stay in the existing Python code
(`storage.py`, `study_tools.py`) and are unchanged.

## Preserved checkpoints

The from-scratch training artefacts remain in `models/` unchanged:
`language_model.pt`, `language_model_pretrained.pt`, `tokenizer.json` and the 15 files
in `models/checkpoints/`. They document the pretraining run but are **not used** for
revision chat in this delivery.

## Tests

```powershell
& ".venv\Scripts\python.exe" -m pytest -q
```

- Baseline before this work: **68 passed (baseline)**. For the current number, see
  `evidence/verification/final-pytest.txt`.
- New/changed tests: `tests/test_ollama_client.py` (19 tests),
  `tests/test_ollama_app.py`, `tests/test_ollama_policy.py` (replacing
  `tests/test_no_ollama.py`).
- If a live-Ollama test is present in the suite, run it only with Ollama running and
  the model pulled; otherwise the live check is the smoke result above.

## Privacy

- No raw data, reviewed teacher data, databases, uploads, API keys or private
  coursework are committed.
- `.gitignore` and `ignore.gitignore` are in sync.
- Revision chat contacts only `127.0.0.1:11434`; the SmolLM2 weights are not
  redistributed in this repository or its ZIP.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Chat errors with "Ollama is not running…" (503) | Start Ollama: `ollama serve` or open the Ollama app; check the `ollama` block in `GET /api/health`. |
| Chat errors with "model is not installed" (503) | `ollama pull smollm2:360m` |
| Port busy on startup | Port 5000: another Flask/app instance — stop it. Port 11434: another Ollama process — quit it, then restart Ollama. |
| Timeout (504) | The model is slow to load or answer; retry, keep `OLLAMA_READ_TIMEOUT` at 90 s. |
| Classifier fallback in use | Expected: trained classifier files are absent, so `rule-fallback-v1` answers. Not an error. |
| "Model not installed" badge in the sidebar | Run `ollama pull smollm2:360m`; the badge refreshes on the next status check. |

## Final ZIP

Path: `C:\Users\rishi\Desktop\Rewiseorist-final.zip`

Contains: source, tests, docs, samples, evidence, this README, `requirements.txt` and
the final model files only (`language_model.pt`, `language_model_pretrained.pt`,
`tokenizer.json`, `model_manifest.json`).

Excluded: intermediate checkpoints, raw data, databases, uploads, secrets, `.venv` and
`teacher_models`. **The SmolLM2 model itself is not in the ZIP** — after extracting,
run `ollama pull smollm2:360m` once.

## Remaining limitations

- The trained performance classifier is absent; `rule-fallback-v1` is used instead.
- SmolLM2 360M is a small model: it may paraphrase and can occasionally fail source
  grounding. Grounding is verified by lexical overlap only, and unverified answers are
  flagged (grounding status is stored with each message and returned by the API).
- External/human/device checks (student panels, device matrix, browser matrix) are
  unverified.
- The from-scratch language model is no longer used for chat in this delivery.
