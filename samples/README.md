# Rewiseorist sample study corpus

## Purpose

This folder contains a small **synthetic, original corpus** written specifically for developing and testing Rewiseorist. It exercises resource import, text extraction, retrieval, quiz display, answer marking, hints, explanations and progress tracking.

The corpus is **not personal data**. It does not describe or come from a real student, and it contains no copied textbook passages. The explanations, examples and questions were written from scratch for this project. You can import it into Rewiseorist for demos and automated tests, but it is a development sample rather than a complete curriculum.

## Contents

| File | Topic | Subject area |
| --- | --- | --- |
| `resources/fractions.md` | Fractions | Maths |
| `resources/forces.md` | Forces | Physics |
| `resources/chemical-bonding.md` | Chemical bonding | Chemistry |
| `resources/indices.md` | Indices | Maths |
| `resources/electric-circuits.md` | Electric circuits | Physics |
| `resources/industrial-revolution.md` | The Industrial Revolution | History |
| `resources/algebra.md` | Algebra | Maths |
| `resources/photosynthesis.md` | Photosynthesis | Biology |
| `resources/particle-model.md` | The particle model | Physics/Chemistry |

Each topic note is 350–600 words and includes headings, definitions, worked examples where they suit the topic, and a short **Key facts** section. The notes are written in plain English for a Year 10/GCSE-style learner.

`quiz_keys.json` contains ten questions for every topic: 90 questions in total. Each topic has five multiple-choice questions and five short-answer questions, spread across easy, medium and hard difficulty.

## Quiz key format

The JSON file is an object whose keys are topic slugs. Each key maps to an array of exactly ten question objects.

| Field | Required for | Meaning |
| --- | --- | --- |
| `id` | All items | Unique identifier, such as `fractions-q1`. |
| `type` | All items | Either `mcq` for multiple choice or `short` for short answer. |
| `question` | All items | The question text shown to the learner. |
| `options` | Multiple choice | Four answer choices. |
| `answer` | All items | The exact text of the correct option for multiple choice; the model answer for short answer. |
| `accepted_keywords` | Short answer | Key terms used when marking; a correct response should include most of the important concepts. Matching is case-insensitive. |
| `hint` | All items | A prompt that helps without giving away the full answer. |
| `explanation` | All items | The reasoning shown after the answer is marked. |
| `difficulty` | All items | `easy`, `medium` or `hard`. |
| `source` | All items | The matching resource filename, such as `fractions.md`. |

The `source` field links each question to its topic note, so an importer can pair the quiz with the correct resource without relying on the JSON key alone. All question IDs are unique across the file.

## Importing into Rewiseorist

1. Start Rewiseorist and open the Resources area.
2. Import the nine Markdown files from `samples/resources/` as study resources.
3. Import `samples/quiz_keys.json` through the quiz or practice import option.
4. Check that each question appears under the topic named by its `source` value.

The sample is intentionally small so imports finish quickly and results are easy to inspect. It is safe to reset or re-import during development.

## Validation

From the repository root, check that the JSON is valid:

```text
python -m json.tool samples/quiz_keys.json
```

A successful check prints the formatted JSON with no error. All content is stored as UTF-8 text.
