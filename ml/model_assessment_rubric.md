# Rewiseorist model assessment rubrics

These rubrics are used to compare candidate teachers and to evaluate the final
student models. They are deliberately separate because revision-answer quality
and performance classification are different tasks.

## 1. Revision answer rubric (12 marks)

| Criterion | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **Accuracy** | Wrong or blank | Mostly wrong | Partly correct | Mostly correct | Fully correct and precise |
| **Relevance** | Off-topic | Tangential | Mostly relevant | Directly relevant | Directly relevant and focused |
| **Clarity** | Unusable | Hard to follow | Understandable | Clear | Clear, well organised and concise |
| **Source use** | No source | Vague claim | Mentions source | Uses source evidence | Uses source evidence and a specific reference |

**Total:** /12

Score meanings:
- **10–12:** Strong teacher example; suitable for distillation after source checking.
- **7–9:** Useful but needs review or correction.
- **4–6:** Weak; only use after substantial correction.
- **0–3:** Reject.

A revision answer must not invent facts. An unsupported claim caps Accuracy at 1
and Source use at 1, even if the prose is fluent.

## 2. Student-performance rubric (6 marks)

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| **Correct category** | Wrong category | Borderline / inconsistent | Correct Weak, Developing or Strong category |
| **Suitable reason** | Missing or irrelevant | Partly explains the evidence | Explains accuracy, trend, hints, independence and difficulty appropriately |
| **Hints and independent answers** | Ignored | Mentioned but not used well | Explicitly considered in the judgement |

**Total:** /6

Score meanings:
- **5–6:** Strong teacher label; suitable for classifier distillation after review.
- **3–4:** Usable but needs review.
- **0–2:** Reject.

The teacher must return a category and a short evidence-based reason. A single
quiz cannot justify a permanent ability label. Fewer than three distinct checked
comparable attempts must produce **Unknown**, not Weak/Developing/Strong.

## 3. Candidate selection rule

For each task, run every candidate at least three times on the same fixed tasks.

1. Average the rubric totals across runs.
2. Prefer candidates with valid structured output and consistent categories.
3. Use latency only as a tie-breaker.
4. Select the top two candidates separately for revision and performance.
5. Record raw outputs, settings, timings and scores in `evidence/ml/`.

## 4. Final model evaluation

- Use held-out examples that were not used for teacher selection, training or
  hyperparameter tuning.
- Keep topics together for the language model and students/personas together for
  the classifier.
- Report actual scores, sample outputs and mistakes.
- Do not tune on the final test set. If the model changes, reserve a fresh test set.
