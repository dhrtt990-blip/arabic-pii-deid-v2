---
base_model: Qwen/Qwen3-1.7B
library_name: peft
pipeline_tag: text-generation
language:
- ar
license: other
license_name: to-be-confirmed-by-the-team
tags:
- lora
- arabic
- pii
- de-identification
- named-entity-recognition
---

# Arabic PII extractor: LoRA adapter for Qwen3-1.7B

**Model card draft for a planned LoRA adapter; training, validation and publication are not yet complete.** The intended adapter will make `Qwen/Qwen3-1.7B` read an Arabic sentence and return personal-information spans as JSON:

```json
{"entities": [{"text": "سلطان الدوسري", "type": "PERS"}, {"text": "مستشفى النخيل", "type": "ORG"}]}
```

Types: `PERS` (person name or kunya), `LOC` (city, district, street, country), `ORG` (named institution), `DATE` (a specific date in words or digits).

The adapter is one part of a de-identification pipeline. National IDs, phone numbers, medical record numbers and e-mail addresses are handled by regex tools, not by this model, and a leakage checker decides whether the redacted text may be released. Code: https://github.com/dhrtt990-blip/arabic-pii-deid-v2.

## Intended use

- Course project (KAUST Academy, NLP Applications 2026): masking personal information in Arabic clinical and administrative text, in Modern Standard Arabic and Saudi dialect.
- Use it inside the pipeline from the repository. On its own it does not detect identifiers and has no guard against a failed or manipulated output.

## Out of scope

- It is not a certified de-identification tool and has not been validated on real patient records. Do not rely on it to release real data.
- Languages other than Arabic, and names written in Latin script, were not a training target.

## How to use (after the adapter has been trained and published)

```python
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE, ADAPTER = "Qwen/Qwen3-1.7B", "owner/replace-with-trained-adapter-id"
tok = AutoTokenizer.from_pretrained(BASE)
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16), ADAPTER)

# SYSTEM_PROMPT is in src/prompts.py of the project repository; use it unchanged.
messages = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "راجع المريض سلطان الدوسري مستشفى النخيل."}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(json.loads(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)))
```

For local serving the adapter is merged and exported to GGUF (`gguf/` in this repository) and run with Ollama; see the project README.

## Training data

Chat-format rows built by `src/data_prep.py`: one system prompt, the sentence as the user turn, the JSON answer as the assistant turn. Sources and licences are in the dataset card (`docs/DATASET_CARD.md`):

- ANERCorp (news, PERS / LOC / ORG);
- an optional Wojood sample (news and other domains; dates);
- seeded template sentences in a clinical and administrative register, with invented names;
- sentences written by the team.

Rows: training and validation data have not yet been prepared; final counts will come from `data/processed/prep_report.json` after preparation.

The frozen Golden Test must remain excluded from training. Before fine-tuning, `src/data_prep.py` will filter exact and near-duplicate overlaps and verify the resulting training set. The preparation report will record the golden-file SHA-256.

## Training procedure

| Setting | Value |
|---|---|
| Method | LoRA (PEFT) with TRL `SFTTrainer`, loss on the assistant answer only |
| Rank / alpha / dropout | 16 / 32 / 0.05, all linear layers |
| Epochs | 2 |
| Learning rate | 1e-4, cosine schedule |
| Batch size | 4, gradient accumulation 4 |
| Max length | 768 tokens |
| Seed | 13 |
| Hardware | Not yet selected; a Google Colab T4 is a candidate |
| Library versions | `requirements-train.lock.txt` in the project repository |

Update this table if the team changes any setting in the notebook.

## Evaluation

Test set: the team's **100-example peer-reviewed Golden Test**, approved and frozen on **17 November 2026** in [commit 469984c00d101512df09851ebd61ea3ef096690e](https://github.com/dhrtt990-blip/arabic-pii-deid-v2/commit/469984c00d101512df09851ebd61ea3ef096690e), SHA-256 `9297f0d21c44ad7f7711af3c254a1e5ff9da1cf863e93162386151993d590459`. Entity matches require exact boundaries and type. Leakage is the share of gold entities not fully masked in released text.

| System | Precision | Recall | F1 | Leakage rate | Invalid JSON |
|---|---|---|---|---|---|
| Untuned Qwen3-1.7B with the full prompt and worked examples | — | — | — | — | — |
| This adapter, model only | — | — | — | — | — |
| This adapter inside the full pipeline | — | — | — | — | — |

Wrong-to-right and right-to-wrong changes between models: not yet evaluated.

Real-model evaluation has not been run. The dashes above indicate unmeasured results, not zero performance. Update this card with verified values from `results/results.md` after evaluation.

## Limitations

- Most training text is news, while the intended input is clinical and administrative. Performance on real clinical notes is unknown.
- Dates written entirely in words and relative dates are weakly covered by the training sources.
- The model can copy an entity with slightly different characters; such strings are not found in the text and are ignored by the pipeline, which lowers recall.
- Real-model safety check on 10 adversarial inputs has not yet been conducted; results will be recorded in `results/safety_report.md`.

## Licence

The adapter's release licence has not yet been selected. The base model is described as Apache 2.0 by its authors; source-data licences must be checked before assigning an adapter licence.
