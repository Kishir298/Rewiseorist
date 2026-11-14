# Rewiseorist Build To-Do List

Ranked by dependency, risk and required order. Test immediately after each file change.
Staged commits/pushes happen throughout; the final GitHub push is the last action.

## Fixed decisions
- [x] **PyStories = TinyStories.** Use `roneneldan/TinyStories` (CDLA-Sharing-1.0). Do not search for or download a separate PyStories dataset.
- [x] **Zero Ollama at any stage.** No Ollama calls for assessment, data generation, training, tests or runtime. Teacher models run through llama.cpp.
- [x] **Small DD model.** About 3M parameters, 6k tokenizer, 4 layers, 128 context. No larger training runs.
- [x] Git-first overrides the Criterion C Git-last step.
- [x] Human visual/navigation/onboarding ratings cannot be fabricated; mark them pending.

## P0 — Safe resume, tracker and baseline commit
- [x] Check local/remote history and preserve `main` at `ba038dc`.
- [x] Verify ignore rules cover secrets, DBs, uploads, raw data, model weights and private screenshots.
- [x] Remove unsafe stray files from the project root.
- [x] Run the full test suite before committing (28 passed).
- [x] Rewrite this ranked tracker for the final plan.
- [ ] Commit and push the tested backend/ML baseline.
- [ ] Keep both ignore files in sync after any ignore change.

## P1 — Requirements and dataset record
- [ ] Audit `docs/requirements-checklist.md` against Criterion A/B/C sources.
- [ ] Record PyStories=TinyStories, licence, checksum and raw-data exclusion in the checklist.
- [ ] Map every requirement to a file/route and a test.
- [ ] List backend gaps before frontend work (flashcards, mock exam, onboarding, media pause, API notes).

## P2 — Backend verification and gap fixes
- [x] Implement `storage.py` schema, profiles, resources, sessions, attempts, plans, rewards, notes, backups.
- [x] Implement `resources.py` limits, extraction, chunking, five-file grouping and retrieval.
- [x] Implement `study_tools.py` marking, quizzes, plans, progress, rewards and store helpers.
- [x] Implement `app.py` routes, validation, errors and inactivity manager.
- [x] Tests for storage/resources/study/app pass.
- [ ] Add any missing routes needed by the six-page frontend.
- [ ] Keep requirement imports in `requirements.txt`; optional OCR stays separate.

## P3 — Teacher runtime without Ollama
- [ ] Install/verify a pinned llama.cpp CPU runtime on Windows.
- [ ] Download two or three small licence-checked instruct GGUF models (about 1–3B) from Hugging Face.
- [ ] Record teacher model names, sources, licences, SHA-256 and runtime version under ignored `teacher_models/` and in `evidence/ml/`.
- [ ] Implement `ml/teacher_client.py` for llama.cpp's OpenAI-compatible endpoint.
- [ ] Implement `ml/assess_candidates.py` with fixed revision and performance tasks, repeated runs and the separate rubrics.
- [ ] Rank candidates separately; select the top two revision and top two performance teachers.
- [ ] Add `tests/test_no_ollama.py` scanning app code for Ollama usage/port 11434.

## P4 — Teacher data generation and review
- [ ] Implement `ml/generate_teacher_data.py` for revision answers/feedback/quizzes and synthetic study histories.
- [ ] Implement `ml/review_teacher_data.py` for deduplication, missing fields, unsupported claims, disagreements and grouped splits.
- [ ] Generate modest reviewed datasets; no leakage by topic/persona.
- [ ] Save `data/teacher/reviewed/{train,val,test}.jsonl` plus review/selection evidence.
- [ ] Re-run review after fixes until clean.

## P5 — Small language model
- [ ] Keep the running TinyStories pretraining run, or stop at the best checkpoint if validation stalls.
- [ ] Record actual parameters, tokens, training time, loss curve, validation perplexity and raw samples.
- [ ] Fine-tune only on reviewed revision data with masked response loss (a few epochs).
- [ ] Save `models/language_model.pt`, tokenizer, config and manifest.
- [ ] Evaluate on held-out examples: perplexity, keyword/reference accuracy, source-citation rate, samples and mistakes.
- [ ] Compare pretrained vs fine-tuned; write a model card with honest limitations.

## P6 — Student-performance classifier
- [ ] Implement `ml/train_character.py` using the same features the app computes.
- [ ] Train on reviewed teacher labels with grouped splits.
- [ ] Compare with majority-class and rule-based baselines.
- [ ] Enforce `Unknown` for fewer than three distinct checked comparable attempts.
- [ ] Save model, preprocessing metadata, metrics, confusion matrix and example mistakes.

## P7 — Local service and real lifecycle
- [ ] Implement `ml/service.py` loading both own models with no network calls.
- [ ] Implement `ml/model_worker.py` as a real subprocess worker.
- [ ] Update `ml/runtime.py`/`app.py`: 2 min unload, 7 min worker stop, 10 min media pause, website stays up, next AI request cold-starts.
- [ ] Implement `ml/evaluate_models.py` for frozen held-out evaluation.
- [ ] Test missing model files return 503, never a fake answer.
- [ ] Verify chat and classification with Ollama stopped and no cloud access.

## P8 — Frontend, all six pages
- [ ] Create `frontend/index.html`: Home, Study, Resources, Progress, Store, Settings; 240px sidebar; 44×44 profile; 48×48 help; accessibility landmarks.
- [ ] Create `stylingscripts/style.css`: ACCESSFM, responsive 360/768/1366/1920, hover ≤150ms, themes/contrast, print, AED 0, no paywalls.
- [ ] Create `stylingscripts/app.js`: onboarding/profiles, prompt transfer, saved chats/sources, uploads/folders, quizzes/mock/flashcards, plans/restore, progress, rewards/store, settings, notes/drawing, help, inactivity/media pause, backups.
- [ ] Test each feature immediately with Flask client and a real browser.
- [ ] Fix dead buttons, empty states, errors and accessibility defects.

## P9 — Workflow and verification evidence
- [ ] Implement `tests/test_workflow.py` for the complete study journey.
- [ ] Run all backend tests and fix failures.
- [ ] Install Playwright Chromium; test at 360/768/1366/1920 and save genuine screenshots.
- [ ] Run 40-text-element contrast audit (at least 39 pass), keyboard/focus, hover timing and print checks.
- [ ] Test 50 saved sessions, reopen sessions, upload limits and five-file grouping.
- [ ] Run scaled inactivity tests plus real 2/7/10-minute observations; confirm site stays available.
- [ ] Test offline launch and clean restart; verify both models load without retraining.
- [ ] Record real metrics, timings, screenshots and failures; fix and rerun affected checks.
- [ ] Mark human-only checks pending.

## P10 — README, ZIP and clean-room proof
- [ ] Write `README.md` with exact Windows setup, training, launch, test, backup commands.
- [ ] Document TinyStories licence/checksum, zero-Ollama policy, model metrics, limitations and unverified items.
- [ ] Create a ZIP with source, both own models, tokenizer, config and instructions; exclude raw data, teacher GGUFs, DBs, uploads, secrets and personal screenshots.
- [ ] Extract to a clean folder, fresh venv, launch offline and verify both models without retraining.
- [ ] Save clean-room results and ZIP checksum; fix packaging defects and repeat.

## P11 — Final audit, commit and push
- [ ] Re-read every checklist item; mark implemented/verified/unverified.
- [ ] Audit dead buttons, missing imports, secrets, personal data, weights/raw data in Git and any Ollama dependency.
- [ ] Create and fix a follow-up audit list for every weakness found.
- [ ] Verify `requirements.txt` includes every imported third-party package.
- [ ] Push reviewed stages normally; preserve remote history; never force-push.
- [ ] Final handoff: launch command, ZIP path/checksum, GitHub branch/commit, actual training times, test results, unverified items.

## Human-only checks (cannot be fabricated)
- [ ] 10/12 student visual-appeal ratings.
- [ ] 10/12 student navigation ratings.
- [ ] 12/12 student onboarding checks.
