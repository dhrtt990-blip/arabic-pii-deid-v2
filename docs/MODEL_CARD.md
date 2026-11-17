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

## How to use

```python
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE, ADAPTER = "Qwen/Qwen3-1.7B", "TBD/this-repository-id"
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

Rows: TBD train / TBD validation (from `data/processed/prep_report.json`).

No golden-test sentence is in the training data: every training sentence that equalled, nearly equalled or contained a golden sentence was removed and the absence of overlap was verified again before training. The report records the SHA-256 of the golden file used.

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
| Hardware | TBD (for example one T4 GPU on Google Colab) |
| Library versions | `requirements-train.lock.txt` in the project repository |

Update this table if the team changes any setting in the notebook.

## Evaluation

Test set: the team's **100-example peer-reviewed Golden Test**, approved and frozen on **17 November 2026** in [commit 469984c00d101512df09851ebd61ea3ef096690e](https://github.com/dhrtt990-blip/arabic-pii-deid-v2/commit/469984c00d101512df09851ebd61ea3ef096690e), SHA-256 `9297f0d21c44ad7f7711af3c254a1e5ff9da1cf863e93162386151993d590459`. Entity matches require exact boundaries and type. Leakage is the share of gold entities not fully masked in released text.

| System | Precision | Recall | F1 | Leakage rate | Invalid JSON |
|---|---|---|---|---|---|
| Untuned Qwen3-1.7B with the full prompt and worked examples | TBD | TBD | TBD | TBD | TBD |
| This adapter, model only | TBD | TBD | TBD | TBD | TBD |
| This adapter inside the full pipeline | TBD | TBD | TBD | TBD | TBD |

Examples that changed from wrong to right and from right to wrong between the untuned and the tuned model: TBD and TBD.

Results are pending. Replace every TBD with the numbers from `results/results.md` and upload this card again.

## Limitations

- Most training text is news, while the intended input is clinical and administrative. Performance on real clinical notes is unknown.
- Dates written entirely in words and relative dates are weakly covered by the training sources.
- The model can copy an entity with slightly different characters; such strings are not found in the text and are ignored by the pipeline, which lowers recall.
- Safety check on 10 adversarial inputs: TBD passed, TBD withheld, TBD failed (`results/safety_report.md`).

## Licence

TBD. The base model is released by its authors under Apache 2.0. Before choosing a licence for this adapter, check the licences of the training sources listed in the dataset card.
