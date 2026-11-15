"""Score three systems on the golden test.

    1. baseline  the untuned model with the full instruction and worked examples
    2. tuned     the LoRA-tuned model on its own
    3. full      the tuned model inside the pipeline (regex tools + leakage checker)

    python -m src.evaluate --base-model qwen3:1.7b --tuned-model pii-qwen3-lora
    python -m src.evaluate --client mock --min-rows 0 --allow-unreviewed   # wiring check only

Reported for each system
    - entity-level precision / recall / F1 (a prediction counts only if its
      boundaries and its type both match the gold entity);
    - leakage rate: gold entities not fully masked in the text that was released;
    - error types: invalid JSON, boundary error, wrong entity type, false
      positive, false negative, each with a count and real examples;
    - examples that changed from wrong to right and from right to wrong between
      systems, because with about 100 examples a small gain can be chance.

Two scopes are scored. "model_types" covers PERS, LOC, ORG and DATE, the types
the model is trained to produce, and is the fair before/after comparison.
"all_types" adds national ID, phone, MRN and e-mail, which only the regex
layer handles, and shows what the system adds around the model.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .common import ALL_TYPES, MODEL_TYPES, Span, coverage_mask, gold_spans, is_fully_covered, read_jsonl, resolve_overlaps, sha256_file, write_jsonl
from .pipeline import DEFAULT_BASE_MODEL, DEFAULT_HOST, DEFAULT_TUNED_MODEL, DeidPipeline, MockClient, OllamaClient, run_model
from .validate_golden import validate_rows

SYSTEMS = ("baseline", "tuned", "full")
SCOPES = {"model_types": MODEL_TYPES, "all_types": ALL_TYPES}
NATIVE_SCOPE = {"baseline": "model_types", "tuned": "model_types", "full": "all_types"}
TRANSITIONS = [("baseline", "tuned", "model_types"), ("tuned", "full", "all_types"), ("baseline", "full", "all_types")]
ERROR_LABELS = ("boundary_error", "wrong_type", "false_positive", "false_negative")
SYSTEM_TITLES = {
    "baseline": "1. Untuned model + prompt",
    "tuned": "2. Tuned model only",
    "full": "3. Full system",
}


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict_model_only(client, example: dict) -> dict:
    """Model alone: whatever it extracts is masked, and the text is released
    as it is. There is no regex layer and no guard."""
    text = example["text"]
    t0 = time.perf_counter()
    ext = run_model(client, text)
    spans = resolve_overlaps(ext.spans) if ext.ok else []
    entities = [s.to_dict(text) for s in spans]
    return {
        "id": example["id"],
        "status": "OK" if ext.ok else ("MODEL_UNAVAILABLE" if ext.unavailable else "INVALID_OUTPUT"),
        "released": True,
        "invalid_json": not ext.ok and not ext.unavailable,
        "model_unavailable": ext.unavailable,
        "cleaned": ext.cleaned,
        "not_in_text": ext.not_in_text,
        "error": ext.error,
        "pred": entities,
        "mask": entities,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


def predict_full(pipeline: DeidPipeline, example: dict) -> dict:
    res = pipeline.run(example["text"])
    return {
        "id": example["id"],
        "status": res.status,
        "released": res.released,
        "invalid_json": res.model.get("ok") is False and not res.model.get("unavailable"),
        "model_unavailable": bool(res.model.get("unavailable")),
        "cleaned": bool(res.model.get("cleaned")),
        "not_in_text": res.model.get("not_in_text", []),
        "error": res.model.get("error"),
        "pred": res.entities,
        "mask": res.entities if res.released else [],
        "elapsed_s": res.elapsed_s,
    }


def _spans(dicts: list[dict], types: tuple[str, ...] | None = None) -> list[Span]:
    return [Span(d["start"], d["end"], d["type"], d.get("source", "model")) for d in dicts if types is None or d["type"] in types]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def classify_example(gold: list[Span], pred: list[Span]) -> dict:
    """Exact matches are true positives. Every other prediction and every other
    gold entity gets one error label."""
    gold_keys = {(g.start, g.end, g.type) for g in gold}
    pred_keys = {(p.start, p.end, p.type) for p in pred}
    exact = gold_keys & pred_keys
    pred_errors, gold_errors = [], []
    for p in pred:
        if (p.start, p.end, p.type) in exact:
            continue
        touching = [g for g in gold if g.overlaps(p)]
        if not touching:
            label = "false_positive"
        elif any(g.type == p.type for g in touching):
            label = "boundary_error"
        else:
            label = "wrong_type"
        pred_errors.append((p, label, touching))
    for g in gold:
        if (g.start, g.end, g.type) in exact:
            continue
        touching = [p for p in pred if p.overlaps(g)]
        if not touching:
            label = "false_negative"
        elif any(p.type == g.type for p in touching):
            label = "boundary_error"
        else:
            label = "wrong_type"
        gold_errors.append((g, label, touching))
    return {"tp": len(exact), "pred_errors": pred_errors, "gold_errors": gold_errors}


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def score_system(golden: list[dict], predictions: dict[str, dict], scope: str) -> dict:
    types = SCOPES[scope]
    tp = fp = fn = 0
    by_type = {t: Counter() for t in types}
    errors = Counter()
    error_rows: list[dict] = []
    per_example: dict[str, dict] = {}
    gold_total = gold_released = leaked = examples_with_leak = withheld = invalid = cleaned = not_in_text = 0

    for ex in golden:
        text = ex["text"]
        pr = predictions[ex["id"]]
        gold = [g for g in gold_spans(ex) if g.type in types]
        pred = _spans(pr["pred"], types)
        result = classify_example(gold, pred)

        tp += result["tp"]
        fp += len(pred) - result["tp"]
        fn += len(gold) - result["tp"]
        exact_gold = {(g.start, g.end, g.type) for g in gold} & {(p.start, p.end, p.type) for p in pred}
        for g in gold:
            by_type[g.type]["tp" if (g.start, g.end, g.type) in exact_gold else "fn"] += 1
        for p in pred:
            if (p.start, p.end, p.type) not in exact_gold:
                by_type[p.type]["fp"] += 1

        for span, label, touching in result["gold_errors"]:
            errors[label] += 1
            error_rows.append({
                "id": ex["id"], "label": label, "side": "gold", "type": span.type, "span_text": text[span.start:span.end],
                "predicted_instead": [{"text": text[t.start:t.end], "type": t.type} for t in touching], "text": text,
            })
        for span, label, touching in result["pred_errors"]:
            if label == "false_positive":
                errors[label] += 1
            error_rows.append({
                "id": ex["id"], "label": label, "side": "prediction", "type": span.type, "span_text": text[span.start:span.end],
                "gold_there": [{"text": text[t.start:t.end], "type": t.type} for t in touching], "text": text,
            })

        invalid += pr["invalid_json"]
        cleaned += pr["cleaned"]
        not_in_text += len(pr.get("not_in_text", []))
        gold_total += len(gold)
        n_leaked = 0
        if pr["released"]:
            gold_released += len(gold)
            mask = coverage_mask(len(text), _spans(pr["mask"]))
            n_leaked = sum(1 for g in gold if not is_fully_covered(g, mask))
        else:
            withheld += 1
        leaked += n_leaked
        examples_with_leak += n_leaked > 0
        per_example[ex["id"]] = {
            "correct": pr["released"] and not result["pred_errors"] and not result["gold_errors"],
            # A withheld example leaks nothing but delivers nothing either, so it is
            # not counted as a success here; it is reported in its own column.
            "released_leak_free": pr["released"] and n_leaked == 0,
        }

    n = len(golden)
    return {
        "scope": scope,
        "examples": n,
        "gold_entities": gold_total,
        **_prf(tp, fp, fn),
        "by_type": {t: _prf(c["tp"], c["fp"], c["fn"]) for t, c in by_type.items()},
        "leaked_entities": leaked,
        "gold_entities_in_released_text": gold_released,
        "leakage_rate": round(leaked / gold_released, 4) if gold_released else 0.0,
        "examples_with_leak": examples_with_leak,
        "examples_withheld": withheld,
        "examples_fully_correct": sum(1 for v in per_example.values() if v["correct"]),
        "errors": {"invalid_json_examples": invalid, **{label: errors[label] for label in ERROR_LABELS}},
        "json_needed_cleaning_examples": cleaned,
        "entities_not_in_text": not_in_text,
        "per_example": per_example,
        "error_rows": error_rows,
    }


def count_transitions(before: dict, after: dict, key: str) -> dict:
    w2r = sorted(i for i in before if not before[i][key] and after[i][key])
    r2w = sorted(i for i in before if before[i][key] and not after[i][key])
    return {
        "wrong_to_right": len(w2r),
        "right_to_wrong": len(r2w),
        "right_in_both": sum(1 for i in before if before[i][key] and after[i][key]),
        "wrong_in_both": sum(1 for i in before if not before[i][key] and not after[i][key]),
        "wrong_to_right_ids": w2r,
        "right_to_wrong_ids": r2w,
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _leak(r: dict) -> str:
    """Leakage rate, or "n/a" when no text was released: 0% would read as a success."""
    return _pct(r["leakage_rate"]) if r["gold_entities_in_released_text"] else "n/a"


def _golden_commit(path: str) -> str | None:
    """Commit that last touched the golden file: the proof it predates training."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%h %cI", "--", path],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
        return out or None
    except (OSError, subprocess.SubprocessError):
        return None


def render_markdown(meta: dict, scores: dict, transitions: list[dict], max_examples: int = 5) -> str:
    L: list[str] = ["# Evaluation on the golden test", ""]
    if meta["client"] == "mock":
        L += ["> **DRY RUN WITH THE MOCK CLIENT.** These numbers come from a rule-based stand-in, not from a model. "
              "They only prove the script runs and must not be reported as results.", ""]
    L += [
        f"- Test set: `{meta['golden_file']}`, {meta['golden_rows']} examples, SHA-256 `{meta['golden_sha256']}`",
        f"- Golden file last commit: {meta['golden_commit'] or 'not available (not a git checkout, or file not committed)'}",
        f"- Date (UTC): {meta['created_utc']}",
        f"- Baseline model: `{meta['models']['baseline']}` | tuned model: `{meta['models']['tuned']}`",
        "- Matching rule: a predicted entity is correct only if its start, end and type equal the gold entity.",
        "- Leakage: a gold entity counts as leaked if any of its characters is unmasked in released text. "
        "The rate is over the gold entities of released examples only; withheld examples release nothing and are counted in their own column.",
        "",
    ]

    def table(scope: str, title: str) -> None:
        rows = [s for s in SYSTEMS if (s, scope) in scores]
        if not rows:
            return
        L.extend([f"## {title}", "",
                  "| System | Precision | Recall | F1 | Leakage in released text | Examples with a leak | Withheld | Invalid JSON | Fully correct |",
                  "|---|---|---|---|---|---|---|---|---|"])
        for s in rows:
            r = scores[(s, scope)]
            L.append(
                f"| {SYSTEM_TITLES[s]} | {_pct(r['precision'])} | {_pct(r['recall'])} | {_pct(r['f1'])} | "
                f"{_leak(r)} ({r['leaked_entities']}/{r['gold_entities_in_released_text']}) | {r['examples_with_leak']} | "
                f"{r['examples_withheld']} | {r['errors']['invalid_json_examples']} | {r['examples_fully_correct']}/{r['examples']} |"
            )
        L.append("")

    table("model_types", "Before and after fine-tuning: PERS, LOC, ORG, DATE")
    table("all_types", "What the system adds: all eight types")

    L += ["## Error types", "",
          "Counts are in each system's own scope (model systems: four types; full system: eight types). "
          "Boundary error, wrong type and false negative are counted per gold entity; false positive per spurious prediction.",
          "",
          "| System | Invalid JSON (examples) | Boundary error | Wrong entity type | False positive | False negative | JSON needed cleaning | Strings not in text |",
          "|---|---|---|---|---|---|---|---|"]
    for s in SYSTEMS:
        key = (s, NATIVE_SCOPE[s])
        if key in scores:
            r = scores[key]
            e = r["errors"]
            L.append(f"| {SYSTEM_TITLES[s]} | {e['invalid_json_examples']} | {e['boundary_error']} | {e['wrong_type']} | "
                     f"{e['false_positive']} | {e['false_negative']} | {r['json_needed_cleaning_examples']} | {r['entities_not_in_text']} |")
    L.append("")

    L += ["## Per-type scores", ""]
    for s in SYSTEMS:
        key = (s, NATIVE_SCOPE[s])
        if key not in scores:
            continue
        L += [f"**{SYSTEM_TITLES[s]}**", "", "| Type | Gold | Precision | Recall | F1 |", "|---|---|---|---|---|"]
        for t, r in scores[key]["by_type"].items():
            L.append(f"| {t} | {r['tp'] + r['fn']} | {_pct(r['precision'])} | {_pct(r['recall'])} | {_pct(r['f1'])} |")
        L.append("")

    if transitions:
        L += ["## Examples that changed", "",
              "An example is right when every entity in scope is found exactly, nothing extra is predicted and text is released.",
              "",
              "| From | To | Scope | Measure | Wrong to right | Right to wrong | Right in both | Wrong in both |",
              "|---|---|---|---|---|---|---|---|"]
        for t in transitions:
            for measure, label in (("correct", "fully correct"), ("released_leak_free", "released with no leak")):
                c = t[measure]
                L.append(f"| {t['from']} | {t['to']} | {t['scope']} | {label} | {c['wrong_to_right']} | {c['right_to_wrong']} | "
                         f"{c['right_in_both']} | {c['wrong_in_both']} |")
        L.append("")

    L += ["## Error examples", "",
          f"The first {max_examples} of each kind in file order, not selected. Every error is listed in `errors_<system>.jsonl`.", ""]
    for s in SYSTEMS:
        key = (s, NATIVE_SCOPE[s])
        if key not in scores:
            continue
        L += [f"### {SYSTEM_TITLES[s]}", ""]
        rows = scores[key]["error_rows"]
        any_error = False
        for label in ERROR_LABELS:
            side = "prediction" if label == "false_positive" else "gold"
            picked = [r for r in rows if r["label"] == label and r["side"] == side][:max_examples]
            if not picked:
                continue
            any_error = True
            L.append(f"**{label.replace('_', ' ')}**")
            L.append("")
            for r in picked:
                other = r.get("predicted_instead") or r.get("gold_there") or []
                other_txt = "; ".join(f"{o['text']} ({o['type']})" for o in other)
                note = f" -> model gave: {other_txt}" if side == "gold" and other_txt else (f" -> gold there: {other_txt}" if other_txt else "")
                L.append(f"- `{r['id']}` {r['type']} «{r['span_text']}»{note}  \n  {r['text']}")
            L.append("")
        if not any_error:
            L += ["No errors of these kinds.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_predictions(system: str, golden: list[dict], args: argparse.Namespace) -> tuple[list[dict], str]:
    if args.client == "mock":
        client = MockClient("weak" if system == "baseline" else "strong")
    else:
        model = args.base_model if system == "baseline" else args.tuned_model
        client = OllamaClient(model, host=args.host, few_shot=(system == "baseline"), use_schema=not args.no_schema)
        ok, why = client.is_available()
        if not ok:
            raise SystemExit(f"{why}. Start Ollama and install the model, or use --client mock for a wiring check.")
    pipeline = DeidPipeline(client) if system == "full" else None
    rows = []
    for i, ex in enumerate(golden, 1):
        row = predict_full(pipeline, ex) if pipeline else predict_model_only(client, ex)
        if row["model_unavailable"]:
            # A timeout or a server error is not a model answer. Scoring it would
            # count it as invalid JSON and as leaked text, so the run stops instead.
            raise SystemExit(f"the model server failed on example {ex['id']} ({row['error']}). "
                             "Nothing was scored. Fix the server and run the evaluation again.")
        rows.append(row)
        if i % 10 == 0 or i == len(golden):
            print(f"  {system}: {i}/{len(golden)}", flush=True)
    return rows, client.name


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate baseline, tuned model and full system on the golden test.")
    p.add_argument("--golden", default="data/golden_test.jsonl")
    p.add_argument("--out", default=None, help="output directory (default: results, or results/dryrun_mock for the mock client)")
    p.add_argument("--client", choices=["ollama", "mock"], default="ollama")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--tuned-model", default=DEFAULT_TUNED_MODEL)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    p.add_argument("--from-predictions", help="re-score saved predictions_<system>.jsonl files from this directory")
    p.add_argument("--no-schema", action="store_true", help="do not constrain the model output to the JSON schema")
    p.add_argument("--min-rows", type=int, default=100)
    p.add_argument("--allow-unreviewed", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="score only the first N examples (debugging)")
    args = p.parse_args(argv)

    try:
        golden = read_jsonl(args.golden)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    errors, _, _ = validate_rows(golden, min_rows=args.min_rows, require_review=not args.allow_unreviewed)
    if errors:
        print("ERROR: golden test is not valid; run `python -m src.validate_golden`:")
        for e in errors[:10]:
            print(f"  {e}")
        return 1
    if args.limit:
        golden = golden[: args.limit]

    out = Path(args.out or ("results/dryrun_mock" if args.client == "mock" else "results"))
    out.mkdir(parents=True, exist_ok=True)

    predictions: dict[str, dict[str, dict]] = {}
    models: dict[str, str] = defaultdict(lambda: "not run")
    for system in args.systems:
        if args.from_predictions:
            rows = read_jsonl(Path(args.from_predictions) / f"predictions_{system}.jsonl")
            models[system] = "from saved predictions"
        else:
            rows, name = run_predictions(system, golden, args)
            models[system] = name
            write_jsonl(out / f"predictions_{system}.jsonl", rows)
        predictions[system] = {r["id"]: r for r in rows}
        missing = [ex["id"] for ex in golden if ex["id"] not in predictions[system]]
        if missing:
            print(f"ERROR: predictions for {system} lack {len(missing)} golden id(s), e.g. {missing[:3]}")
            return 1

    scores: dict[tuple[str, str], dict] = {}
    for system in args.systems:
        for scope in SCOPES:
            scores[(system, scope)] = score_system(golden, predictions[system], scope)
        write_jsonl(out / f"errors_{system}.jsonl", scores[(system, NATIVE_SCOPE[system])]["error_rows"])

    transitions = []
    for a, b, scope in TRANSITIONS:
        if (a, scope) in scores and (b, scope) in scores:
            pa, pb = scores[(a, scope)]["per_example"], scores[(b, scope)]["per_example"]
            transitions.append({"from": a, "to": b, "scope": scope,
                                "correct": count_transitions(pa, pb, "correct"),
                                "released_leak_free": count_transitions(pa, pb, "released_leak_free")})

    latency = {s: round(sum(r.get("elapsed_s", 0) for r in predictions[s].values()) / len(golden), 3) for s in args.systems}
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "client": args.client,
        "golden_file": args.golden,
        "golden_rows": len(golden),
        "golden_sha256": sha256_file(args.golden),
        "golden_commit": _golden_commit(args.golden),
        "models": {s: models[s] for s in SYSTEMS},
        "mean_seconds_per_example": latency,
    }
    slim = {f"{s}/{scope}": {k: v for k, v in r.items() if k not in ("per_example", "error_rows")} for (s, scope), r in scores.items()}
    (out / "results.json").write_text(json.dumps({"meta": meta, "scores": slim, "transitions": transitions}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "results.md").write_text(render_markdown(meta, scores, transitions), encoding="utf-8")

    for system in args.systems:
        r = scores[(system, NATIVE_SCOPE[system])]
        print(f"{system:<9} [{NATIVE_SCOPE[system]}]  P {_pct(r['precision'])}  R {_pct(r['recall'])}  F1 {_pct(r['f1'])}  "
              f"leak {_leak(r)}  withheld {r['examples_withheld']}  invalid JSON {r['errors']['invalid_json_examples']}")
    print(f"report: {out / 'results.md'}")
    if args.client == "mock":
        print("NOTE: mock client. These are not project results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
