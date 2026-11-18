# Rewiseorist — Delivery Status

Short, honest status for this delivery. Baseline test result before this work:
**68 passed (baseline)**; current number is recorded in
`evidence/verification/final-pytest.txt`.

## Completed

- [x] **SmolLM2 360M pulled and smoke-tested.** `smollm2:360m` (tag `297281b699fc`,
  ~725 MB) pulled 2026-11-18; live smoke returned "OK" in 3.73 s
  (`evidence/verification/ollama-smoke.json`; Ollama 0.40.16).
- [x] **Ollama client.** `ml/ollama_client.py`: typed failures
  (`OllamaUnavailable`/`OllamaModelMissing`/`OllamaTimeout`/`OllamaEmptyAnswer`/
  `OllamaHTTPError`), top 3 passages capped at 600 chars, lexical-overlap grounding,
  fabricated citations stripped; prompts in `build_ollama_revision_messages`
  (`ml/prompts.py`).
- [x] **Flask integration.** `OLLAMA_*` config keys in `app.py`; `/api/chat` local mode
  calls the Ollama client; `/api/models/status` and `/api/health` include an `ollama`
  status block; `/api/models/warmup` and `/api/models/unload` manage Ollama and the
  classifier worker.
- [x] **Error handling.** Ollama failures mapped to clear HTTP 502/503/504 messages
  telling the user exactly what to run.
- [x] **Classifier fallback.** Trained classifier files absent → `/api/classify` uses
  `rule-fallback-v1` (`rule_label` in `ml/train_character.py`), `fallback=true`.
- [x] **Frontend status.** Model badge/sidebar show SmolLM2 360M ↔ Ollama state
  (ready / pull needed / offline), with warm-up and unload controls.
- [x] **Tests.** `tests/test_ollama_client.py` (19 tests), `tests/test_ollama_app.py`,
  `tests/test_ollama_policy.py` (replacing `tests/test_no_ollama.py`).
- [x] **README.** `README.md` with requirements, install, exact launch/stop commands,
  config table, chat flow, fallback, privacy, troubleshooting, ZIP and limitations.
- [x] **Final ZIP.** `C:\Users\rishi\Desktop\Rewiseorist-final.zip` — source, tests,
  docs, samples, evidence, README, requirements and final model files only.

## Remaining gaps

- [ ] **Trained performance classifier.** `models/performance_model.joblib` and
  `performance_preprocessor.json` are absent; the rule fallback stays in use until
  training runs and its artefacts are saved.
- [ ] **Live-evidence pointer.** Keep `evidence/verification/` current (final pytest
  run, timings, smoke results) — `final-pytest.txt` is the source of truth for the
  current test count.
- [ ] **Human/device checks.** Student panels, onboarding, keyboard/screen-reader,
  device and browser matrices remain unverified (cannot be fabricated).
- [ ] **Older UI backlog.** Resources tabs/charts, print and editor workflows in
  `frontend/index.html` / `stylingscripts/app.js`.
- [ ] **`tests/test_workflow.py`** — full study-journey test not written.

## Known, documented limitations

- SmolLM2 360M is small: it may paraphrase and can occasionally fail source grounding
  (grounding checked by lexical overlap; status stored with each message).
- The from-scratch LM checkpoints in `models/` are preserved as training artefacts and
  are no longer used for revision chat in this delivery.
