# Rewiseorist Build To-Do List

Ranked by dependency and risk. Work top-to-bottom; test immediately after each file change.
Final GitHub push is the last action, but staged commits/pushes happen throughout.

## P0 — Repository safety and baseline
- [x] Clone the existing GitHub remote and preserve its history (remote is empty; `main` created).
- [x] Create `ignore.gitignore` and portable `.gitignore`.
- [x] Configure local `core.excludesFile=ignore.gitignore`.
- [x] Verify secrets, databases, uploads, raw data, caches and generated models are ignored.
- [x] Push initial repository baseline and verify GitHub authentication.
- [ ] Keep every later commit reviewed; never force-push.

## P1 — Requirements and architecture
- [ ] Extract Criterion A, B and C requirements into `docs/requirements-checklist.md`.
- [ ] Map every requirement to a file/route and a verification.
- [ ] Record approved changes: single HTML file, separate rubrics, multi-teacher distillation, Git first.
- [ ] Define SQLite schema, API routes, model formats and inactivity state machine.
- [ ] Add this checklist to the repository.

## P2 — Backend foundation
- [ ] Create `requirements.txt` and Python virtual environment; install dependencies.
- [ ] Create project directory structure.
- [ ] Implement `storage.py` schema, migrations, CRUD, plan versions, rewards, backups.
- [ ] Implement `resources.py` upload limits, extraction, chunking, retrieval index, five-file grouping.
- [ ] Implement `study_tools.py` quizzes, marking, feedback, plans, progress, rewards/store rules.
- [ ] Implement `app.py` Flask routes, validation, permissions, error handling.
- [ ] Test each module immediately; fix failures.
- [ ] Commit and push backend stage.

## P3 — ML data and teacher distillation
- [ ] Write `ml/model_assessment_rubric.md`.
- [ ] Implement `ml/assess_candidates.py`; run candidates locally across repeated tasks.
- [ ] Select top two revision teachers and top two performance teachers; save evidence.
- [ ] Implement `ml/generate_teacher_data.py`; generate revision examples and study histories.
- [ ] Implement `ml/review_teacher_data.py`; flag duplicates, missing data, unsupported claims, disagreements.
- [ ] Review accepted data; split by topic and by student/persona without leakage.
- [ ] Commit and push data-pipeline code (not raw datasets).

## P4 — Language model from scratch
- [ ] Verify TinyStories source, format, licence (CDLA-Sharing-1.0) and record checksum.
- [ ] Download a manageable real TinyStories subset; exclude raw data from Git.
- [ ] Implement `ml/tokenizer.py` byte-level BPE from scratch; save vocabulary/special tokens.
- [ ] Implement `ml/language_model.py` decoder-only causal Transformer from scratch.
- [ ] Implement `ml/train_language.py`; run a short timed pilot to choose a CPU budget.
- [ ] Complete the chosen pretraining run; save checkpoints, loss curve, samples, actual timing.
- [ ] Fine-tune on reviewed revision examples from both teachers.
- [ ] Save tokenizer/config/checkpoint and manifest; record evaluation samples/mistakes.
- [ ] Commit and push training code and metrics (not weights).

## P5 — Student-performance classifier
- [ ] Implement `ml/train_character.py` with coursework features.
- [ ] Compare against majority-class and rule-based baselines.
- [ ] Train on reviewed teacher labels with grouped splits.
- [ ] Save model, preprocessing settings, metrics and confusion matrix.
- [ ] Implement the `Unknown` rule for fewer than three distinct checked comparable attempts.
- [ ] Commit and push classifier code and metrics.

## P6 — Local model service and lifecycle
- [ ] Implement `ml/service.py` to load both saved models and return model versions.
- [ ] Implement `ml/model_worker.py` local worker for unload/shutdown/cold-start behaviour.
- [ ] Implement `ml/evaluate_models.py`; evaluate frozen models on untouched held-out examples.
- [ ] Verify the app works with Ollama stopped and no cloud API.
- [ ] Commit and push service/evaluation code.

## P7 — Frontend shell and all six pages
- [ ] Create `frontend/index.html` with Home, Study, Resources, Progress, Store, Settings.
- [ ] Create `stylingscripts/style.css` with ACCESSFM styling, responsiveness, themes, contrast, print.
- [ ] Create `stylingscripts/app.js` navigation, transfers, uploads, chats, quizzes, plans, charts, store, settings.
- [ ] Add notes/drawing canvas, source references, help assistant, profile button.
- [ ] Implement 2/7/10-minute inactivity behaviour and media pause.
- [ ] Test each UI feature immediately; fix failures.
- [ ] Commit and push frontend stage.

## P8 — Full integration and verification
- [ ] Implement `test_workflow.py` complete study journey.
- [ ] Run backend tests for storage, resources, study tools, models, limits, rewards, plans, inactivity.
- [ ] Run Playwright browser tests at 360/768/1366/1920 px; save screenshots.
- [ ] Run contrast, keyboard, print, upload-limit and 50-saved-session checks.
- [ ] Run real/scaled 2/7/10-minute inactivity checks.
- [ ] Run offline and clean-restart checks.
- [ ] Record actual model scores, samples, mistakes and test evidence.
- [ ] Fix all failures and rerun affected checks.
- [ ] Commit and push verification stage.

## P9 — Documentation, packaging and clean-room test
- [ ] Write `README.md` with exact Windows commands for setup, launch, training, testing and backups.
- [ ] Document limitations, optional OCR/cloud/API status, TinyStories licence/checksum.
- [ ] Create `package.py`/ZIP with source, both models, tokenizer, instructions and evidence; exclude personal data.
- [ ] Extract ZIP to a clean folder; fresh venv; launch with Ollama stopped; verify both models without retraining.
- [ ] Save clean-room screenshots/results and checksum.
- [ ] Fix packaging defects and repeat until clean.
- [ ] Commit and push packaging stage.

## P10 — Final audit
- [ ] Re-read every requirement in `docs/requirements-checklist.md`; mark implemented/verified/unverified.
- [ ] Audit for missing features, dead buttons, secrets, personal data, large ignored files.
- [ ] Create a follow-up audit/fix list for every weakness found; fix and retest.
- [ ] Verify `requirements.txt` includes every imported third-party package.
- [ ] Final reviewed commit and push to `origin main` (or report exact blocker).
- [ ] Produce final handoff: launch command, ZIP path, branch/commit, test results, actual run times, unverified items.

## Human-only checks (cannot be fabricated)
- [ ] 10/12 student visual-appeal ratings — requires real participants.
- [ ] 10/12 student navigation ratings — requires real participants.
- [ ] 12/12 student onboarding checks — requires real participants.
