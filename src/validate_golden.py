"""Strict format check for the golden test.

    python -m src.validate_golden data/golden_test.jsonl

Run it before committing the golden test and again before every evaluation.
It exits with status 1 on any error, so it can gate a commit or a CI job.

Row format (one JSON object per line):
    {"id": "g001", "text": "...", "entities": [{"text": "...", "type": "PERS"}],
     "dialect": "msa", "category": "clinical", "written_by": "A", "checked_by": "B"}
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from . import regex_tools
from .common import ALL_TYPES, GoldFormatError, coverage_mask, gold_spans, is_fully_covered, normalize_for_dedup, read_jsonl

PLACEHOLDER_NAMES = {"", "tbd", "todo", "?", "-"}
MAX_TEXT_CHARS = 1000


def validate_rows(rows: list[dict], min_rows: int = 100, require_review: bool = True) -> tuple[list[str], list[str], dict]:
    """Return (errors, warnings, stats)."""
    errors: list[str] = []
    warnings: list[str] = []
    type_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_texts: dict[str, str] = {}
    empty_rows = 0

    if len(rows) < min_rows:
        errors.append(f"only {len(rows)} rows; at least {min_rows} are required")

    for n, row in enumerate(rows, 1):
        rid = row.get("id") if isinstance(row, dict) else None
        label = rid if isinstance(rid, str) and rid else f"row {n}"
        if not isinstance(row, dict):
            errors.append(f"{label}: not a JSON object")
            continue
        if not isinstance(rid, str) or not rid.strip():
            errors.append(f"{label}: missing 'id'")
        elif rid in seen_ids:
            errors.append(f"{label}: duplicate id")
        else:
            seen_ids.add(rid)

        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{label}: missing 'text'")
            continue
        if len(text) > MAX_TEXT_CHARS:
            errors.append(f"{label}: text longer than {MAX_TEXT_CHARS} characters")
        norm = normalize_for_dedup(text)
        if norm in seen_texts:
            errors.append(f"{label}: same text as {seen_texts[norm]}")
        else:
            seen_texts[norm] = label

        entities = row.get("entities")
        if not isinstance(entities, list):
            errors.append(f"{label}: 'entities' must be a list (use [] when there is nothing sensitive)")
            continue
        bad_entity = False
        for ent in entities:
            if not isinstance(ent, dict) or not isinstance(ent.get("text"), str):
                errors.append(f"{label}: each entity needs a 'text' string")
                bad_entity = True
            elif ent["text"] != ent["text"].strip():
                errors.append(f"{label}: entity {ent['text']!r} has leading or trailing spaces")
                bad_entity = True
            elif ent.get("type") not in ALL_TYPES:
                errors.append(f"{label}: entity {ent['text']!r} has unknown type {ent.get('type')!r}; allowed: {', '.join(ALL_TYPES)}")
                bad_entity = True
        if bad_entity:
            continue
        try:
            spans = gold_spans(row)
        except GoldFormatError as exc:
            errors.append(str(exc))
            continue
        if not spans:
            empty_rows += 1
        for s in spans:
            type_counts[s.type] += 1

        # A pattern the regex tools recognise but nobody annotated is usually a missed label.
        mask = coverage_mask(len(text), spans)
        for rs in regex_tools.find_all(text):
            if not is_fully_covered(rs, mask):
                warnings.append(f"{label}: {text[rs.start:rs.end]!r} looks like {rs.type} but is not annotated")

        if require_review:
            writer = str(row.get("written_by", "")).strip()
            checker = str(row.get("checked_by", "")).strip()
            if writer.lower() in PLACEHOLDER_NAMES or checker.lower() in PLACEHOLDER_NAMES:
                errors.append(f"{label}: 'written_by' and 'checked_by' must name the two team members")
            elif writer == checker:
                errors.append(f"{label}: must be checked by someone other than its writer")

    n_ok = len(rows)
    share_empty = empty_rows / n_ok if n_ok else 0.0
    if n_ok and not 0.10 <= share_empty <= 0.30:
        warnings.append(f"{share_empty:.0%} of rows have no entities; aim for 10-30% so false positives can be measured")
    for etype in ALL_TYPES:
        if n_ok and type_counts[etype] == 0:
            warnings.append(f"no example of type {etype}")

    stats = {
        "rows": len(rows),
        "rows_without_entities": empty_rows,
        "entities": sum(type_counts.values()),
        "entities_by_type": {t: type_counts[t] for t in ALL_TYPES},
        "dialects": dict(Counter(str(r.get("dialect", "unspecified")) for r in rows if isinstance(r, dict))),
    }
    return errors, warnings, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the golden test file.")
    parser.add_argument("path", nargs="?", default="data/golden_test.jsonl")
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--allow-unreviewed", action="store_true", help="skip the written_by / checked_by check (drafting only)")
    args = parser.parse_args(argv)

    try:
        rows = read_jsonl(args.path)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    errors, warnings, stats = validate_rows(rows, min_rows=args.min_rows, require_review=not args.allow_unreviewed)

    print(f"File: {args.path}")
    print(f"Rows: {stats['rows']}  (without entities: {stats['rows_without_entities']})")
    print(f"Entities: {stats['entities']}  by type: {stats['entities_by_type']}")
    print(f"Dialects: {stats['dialects']}")
    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    if errors:
        print("FAILED")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
