# Arabic PII De-identification

A small language model, fine-tuned with LoRA, finds personal information in Arabic clinical and administrative text. It works inside a pipeline with deterministic regex tools and a leakage checker, and runs locally in a Streamlit app with right-to-left display.

Course project for *NLP Applications* (KAUST Academy, 2026), topic 6: own project, option A.

> **Status.** The code, the unit tests and the documentation are in place. The golden test, the trained adapter and every result still have to be produced by the team. Nothing in this repository is a result until the Results section below is filled in from `results/results.md`.

## How the project meets the five requirements

| Requirement | Where |
|---|---|
| Fine-tuning with LoRA or QLoRA | `notebooks/train_qwen3_lora_colab.ipynb` trains an adapter on `Qwen/Qwen3-1.7B` and publishes it on the Hugging Face Hub with `docs/MODEL_CARD.md` |
| Evaluation before, after, and for the full system | `src/evaluate.py` scores the untuned model with a full prompt, the tuned model alone, and the tuned model inside the pipeline, all on the same golden test |
| The model works inside a pipeline | `src/pipeline.py`: LLM extractor, regex tools, merger and redactor, leakage checker with a fallback |
| At most 4B parameters | Qwen3-1.7B |
| Local deployment | The tuned model is served by Ollama on the team's machine; `app/streamlit_app.py` is the demo |

## Architecture

```
            ┌───────────────────────────────┐
 text ────► │ 1. LLM extractor (Qwen3-1.7B) │  PERS, LOC, ORG, DATE as JSON
   │        └───────────────────────────────┘
   │        ┌───────────────────────────────┐
   └──────► │ 2. Regex tools                │  national ID / Iqama, mobile, MRN, e-mail, numeric dates
            └───────────────────────────────┘
                          │
            ┌───────────────────────────────┐
            │ 3. Merger and redactor        │  overlaps resolved, spans replaced by [اسم] [هوية] [جوال] ...
            └───────────────────────────────┘
                          │
            ┌───────────────────────────────┐
            │ 4. Leakage checker            │  regex tools run again on the output; masked strings must not reappear
            └───────────────────────────────┘
                          │
              released text, or a blocked status
```

What happens when something fails:

| Status | Meaning | Text released |
|---|---|---|
| `OK` | Model output valid, checker found nothing | yes |
| `OK_AFTER_REPAIR` | Checker found a masked string elsewhere in the text, masked it, and the second check passed | yes |
| `BLOCKED_MODEL_FAILURE` | Model unreachable or its output was not valid JSON. A regex-only preview is shown for human review | no |
| `BLOCKED_LEAK` | A sensitive pattern survived redaction and could not be repaired | no |
| `BLOCKED_UNLOCATED_ENTITY` | The model named a sensitive string that could not be placed on any word of the text: it altered the name while copying it, or the name is glued to another word. The real one may still be visible | no |
| `BLOCKED_INPUT_TOO_LONG` | Input above 20,000 characters | no |

The regex tools do not depend on the model, so a prompt injection that makes the model return nothing still leaves identifiers masked.

## Quick start from a clean clone (no GPU, no model)

Python 3.11 or newer.

```bash
git clone <your-repository-url>
cd <your-repository>
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m pytest -q                  # unit tests
python -m src.evaluate --client mock --golden data/golden_test_template.jsonl --min-rows 0 --allow-unreviewed
python -m src.safety_check --client mock
streamlit run app/streamlit_app.py   # then choose the dry-run engine in the sidebar
```

On Windows the same steps are in `run_local.bat`: double-click it. It creates `.venv`, installs the requirements, runs the tests and the dry runs, checks whether Ollama is installed, and offers to download `qwen3:1.7b` and run the untuned model on the 10 template sentences as a smoke test. Everything is logged to `logs\local_run.txt`. The batch file was written on Linux and has not been run on Windows yet.

`--client mock` and the dry-run engine use a rule-based stand-in, not a model. They only prove the code runs; their reports are written to `results/dryrun_mock/` and are marked as dry runs.

## Full workflow

The order matters: the golden test is fixed and committed before any training data is built.

1. **Write the golden test.** Follow `docs/annotation_guidelines.md`. Start from `data/golden_test_template.jsonl` (10 example rows) and write about 100 rows into `data/golden_test.jsonl`. Every row is written by one member and checked by another.
   ```bash
   python -m src.validate_golden data/golden_test.jsonl
   git add data/golden_test.jsonl && git commit -m "Golden test v1 (fixed before training)"
   ```
2. **Optional training material.** Put Wojood CoNLL files in `data/raw/wojood/` and team-written training sentences in `data/team_synthetic.jsonl` (same row format as the golden test). Never reuse golden sentences.
3. **Train.** Open `notebooks/train_qwen3_lora_colab.ipynb` in Google Colab with a T4 GPU. It validates the golden test, builds the training data with `src/data_prep.py` (which removes and then verifies zero overlap with the golden test), trains the adapter, publishes it, and exports a GGUF file.
4. **Serve the tuned model locally.**
   ```bash
   ollama pull qwen3:1.7b
   python scripts/make_modelfile.py --gguf models/pii-qwen3-1.7b-q8_0.gguf
   ollama create pii-qwen3-lora -f Modelfile
   ```
5. **Evaluate.**
   ```bash
   python -m src.evaluate --base-model qwen3:1.7b --tuned-model pii-qwen3-lora
   ```
   Output in `results/`: `results.md`, `results.json`, per-system predictions and every error with its sentence.
6. **Safety check.**
   ```bash
   python -m src.safety_check --model pii-qwen3-lora
   ```
   Output: `results/safety_report.md`, one row per adversarial input.
7. **Demo.** `streamlit run app/streamlit_app.py`. Record a short screen capture as a fallback for the presentation.

## Repository layout

```
app/streamlit_app.py                    demo: original text, entities and their source, redacted text, checker status, before/after view
src/common.py                           entity types, Arabic text utilities, span matching, model-output parsing
src/regex_tools.py                      national ID, phone, MRN, e-mail, numeric dates
src/prompts.py                          system prompt, baseline few-shot examples, JSON schema
src/pipeline.py                         Ollama client, mock client, pipeline, leakage checker, fallback
src/data_prep.py                        training data in chat format + zero-overlap check against the golden test
src/validate_golden.py                  strict format check for the golden test
src/evaluate.py                         three systems, P/R/F1, leakage, error types, changed examples
src/safety_check.py                     10 adversarial inputs and what the app did with each
scripts/make_modelfile.py               Ollama Modelfile for the exported GGUF
notebooks/train_qwen3_lora_colab.ipynb  LoRA training, Hub upload, GGUF export
data/golden_test_template.jsonl         10 example rows showing the format
data/safety_inputs.jsonl                the 10 adversarial inputs
docs/annotation_guidelines.md           how to write and check golden-test rows (Arabic)
docs/DATASET_CARD.md, docs/MODEL_CARD.md
tests/                                  unit tests
requirements.txt, requirements.lock.txt, requirements-train.txt
```

## Results

Not available yet. After step 5, copy the two tables from `results/results.md` here, with the line that names the test set.

Test set: `data/golden_test.jsonl`, TBD examples, SHA-256 TBD, committed on TBD.

| System | Precision | Recall | F1 | Leakage rate | Invalid JSON |
|---|---|---|---|---|---|
| 1. Untuned Qwen3-1.7B + prompt | TBD | TBD | TBD | TBD | TBD |
| 2. Tuned model only | TBD | TBD | TBD | TBD | TBD |
| 3. Full system | TBD | TBD | TBD | TBD | TBD |

Examples that changed between systems (wrong to right, right to wrong): TBD.
Safety check: TBD passed, TBD withheld, TBD failed of 10.

## Team

| Member | Part |
|---|---|
| TBD | Golden test and annotation guidelines |
| TBD | Data preparation and fine-tuning |
| TBD | Pipeline, regex tools and app |
| TBD | Evaluation, error analysis and safety check |

## Data and licences

Details and the overlap proof are in `docs/DATASET_CARD.md`. In short: ANERCorp from the Hugging Face Hub, an optional Wojood sample, seeded template sentences, and sentences written by the team. The golden test is written by the team and contains invented people and numbers only. No real patient data and no data from any institution is used anywhere in this project.

## Known limitations

- Digits written as words ("واحد صفر ...") are not caught by any regex.
- The training text is mostly news (ANERCorp, Wojood), while the golden test is clinical and administrative. Expect a domain gap, and report it.
- Names in Latin script are not a target of the training data.
- Entity strings are located as whole words. A particle or kunya word glued in front ("لمحمد", "أبومحمد") and a written case ending ("محمدًا") are tolerated, and diacritics, tatweel and extra spaces are ignored. A name inside an unrelated longer word is left alone ("الرياض" in "الرياضة").
- The same rule cannot tell a name from an identical ordinary word. Once "أمل" is masked as a name, "وأمل" ("and he hoped") in the same text is masked too, and a short name is cut out of a longer one that starts with و, ف, ب or ل ("سام" inside "وسام").
- When the model names a string that cannot be placed on a word of the text, the output is withheld. With a model that often alters what it copies, this raises the number of withheld texts; the evaluation reports that number.
- Numeric dates with a two-digit year are matched only with "/" ("12/8/30"), so a rare three-part measurement written that way is masked as a date, and "12-8-30" is not treated as a date.
- Ten digits written one by one ("1 0 4 5 ...") are treated as an ID only when a word such as "هوية" or "إقامة" stands just before them.
- A mobile number written without its leading zero and without a country code ("555123456") is not matched.
- The national ID pattern does not verify the checksum, so some ten-digit numbers that are not IDs are masked.
- A single sentence longer than 1,200 characters is cut at a fixed position before it is sent to the model.
- This is a course project, not a certified de-identification tool.

## What was and was not verified when this code was written

Verified by running it: the unit tests; `src/data_prep.py` on the Wojood sample files; the evaluation script, the safety check and the app with the mock client; the app's HTML escaping.

Not verified: calls against a real Ollama server; the Colab notebook (no GPU was available); the ANERCorp download and its column names; the GGUF export; any model quality.

## Acknowledgements and AI assistance

Data: ANERCorp (Benajiba et al., 2007; CAMeL Lab split) and Wojood (Jarrar et al., 2022). Base model: Qwen3-1.7B.

The code scaffold in this repository was drafted with an AI assistant (Claude) and reviewed by the team. TBD: describe the extent of AI assistance as the course policy requires.
