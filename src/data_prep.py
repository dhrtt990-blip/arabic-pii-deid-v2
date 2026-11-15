"""Build the instruction-tuning data and prove it is disjoint from the golden test.

    python -m src.data_prep \
        --golden data/golden_test.jsonl \
        --wojood data/raw/wojood/train.txt data/raw/wojood/val.txt data/raw/wojood/test.txt \
        --anercorp-hf asas-ai/ANERCorp \
        --team-jsonl data/team_synthetic.jsonl \
        --out data/processed

Sources
    Wojood sample   CoNLL file, one token per line followed by one or more BIO tags (nested).
    ANERCorp        token/tag rows, from the Hugging Face Hub or a local two-column file.
    Team sentences  JSONL in the golden-test row format, written by the team.
    Templates       seeded synthetic clinical / administrative sentences (invented names only).

Guarantee
    The golden test must exist first. Any training sentence that equals, nearly
    equals or contains a golden sentence is removed, the result is verified
    again, and the script fails if a single overlap remains. The report written
    next to the data records the golden file's SHA-256 and what was removed.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import regex_tools
from .common import (
    MODEL_TYPES,
    GoldFormatError,
    Span,
    char_shingles,
    gold_spans,
    normalize_for_dedup,
    read_jsonl,
    resolve_overlaps,
    sha256_file,
    write_jsonl,
)
from .prompts import FEW_SHOT_EXAMPLES, build_training_example
from .validate_golden import validate_rows

WOJOOD_TYPE_MAP = {"PERS": "PERS", "GPE": "LOC", "LOC": "LOC", "FAC": "LOC", "ORG": "ORG", "DATE": "DATE"}
ANERCORP_TYPE_MAP = {"PERS": "PERS", "PER": "PERS", "LOC": "LOC", "ORG": "ORG"}  # MISC is dropped

_NO_SPACE_BEFORE = set(".,،؛;:!؟?)]}»%")
_NO_SPACE_AFTER = set("([{«")
_SENTENCE_FINAL = {".", "!", "؟", "?"}
_DOC_SEPARATOR = "$#$"
MIN_TOKENS, MAX_CHARS = 3, 600


# --------------------------------------------------------------------------- #
# Token/tag data -> text + entities
# --------------------------------------------------------------------------- #
def detokenize(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join tokens into running text and return each token's character offsets."""
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    pos = 0
    prev = None
    for tok in tokens:
        if parts and tok not in _NO_SPACE_BEFORE and prev not in _NO_SPACE_AFTER:
            parts.append(" ")
            pos += 1
        offsets.append((pos, pos + len(tok)))
        parts.append(tok)
        pos += len(tok)
        prev = tok
    return "".join(parts), offsets


_DATE_MONTH_WORDS = set(
    "يناير فبراير مارس أبريل ابريل مايو يونيو يوليو أغسطس اغسطس سبتمبر أكتوبر اكتوبر نوفمبر ديسمبر "
    "محرم صفر ربيع جمادى رجب شعبان رمضان شوال "
    "كانون شباط آذار اذار نيسان أيار ايار حزيران تموز آب أيلول ايلول تشرين".split()
)


def is_specific_date(text: str) -> bool:
    """True for the dates the annotation guidelines count: a numeric date, or a
    month name together with a number. Source corpora also tag vague expressions
    ("اليوم", a year on its own) as DATE; those are not training targets."""
    if regex_tools.find_numeric_dates(text):
        return True
    has_month = any(tok.lstrip("وبلف") in _DATE_MONTH_WORDS or tok in _DATE_MONTH_WORDS for tok in text.split())
    return has_month and any(ch.isdigit() for ch in text)


def tagged_sentence_to_example(tokens: list[str], tags: list[list[str]], type_map: dict[str, str], source: str) -> dict:
    """Convert one BIO-tagged sentence (possibly with several tags per token) to
    {"text", "entities", "source", "vague_dates_dropped"}.

    Each entity type is decoded independently, which recovers nested entities;
    nested and overlapping mentions are then reduced to the outermost span,
    because that is what has to be masked.
    """
    text, offsets = detokenize(tokens)
    open_at: dict[str, int] = {}
    spans: list[Span] = []

    def close(raw_type: str, last_token: int) -> None:
        first_token = open_at.pop(raw_type)
        mapped = type_map.get(raw_type)
        if mapped:
            spans.append(Span(offsets[first_token][0], offsets[last_token][1], mapped, source))

    for i, token_tags in enumerate(tags):
        present: set[str] = set()
        for tag in token_tags:
            if "-" not in tag:
                continue
            prefix, raw_type = tag.split("-", 1)
            if prefix == "B":
                if raw_type in open_at:
                    close(raw_type, i - 1)
                open_at[raw_type] = i
            elif prefix == "I" and raw_type not in open_at:
                open_at[raw_type] = i  # tolerate an I- tag with no B- before it
            if prefix in ("B", "I"):
                present.add(raw_type)
        for raw_type in [t for t in open_at if t not in present]:
            close(raw_type, i - 1)
    for raw_type in list(open_at):
        close(raw_type, len(tokens) - 1)

    kept = [s for s in spans if s.type != "DATE" or is_specific_date(text[s.start:s.end])]
    entities = [{"text": text[s.start:s.end], "type": s.type} for s in resolve_overlaps(kept)]
    return {"text": text, "entities": entities, "source": source, "vague_dates_dropped": len(spans) - len(kept)}


def read_conll_sentences(path: str | Path) -> list[tuple[list[str], list[list[str]]]]:
    """Read `token tag [tag ...]` lines. Blank lines and `$#$` end a sentence."""
    sentences: list[tuple[list[str], list[list[str]]]] = []
    tokens: list[str] = []
    tags: list[list[str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if not parts or parts[0] == _DOC_SEPARATOR:
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
                continue
            tokens.append(parts[0])
            tags.append(parts[1:] or ["O"])
    if tokens:
        sentences.append((tokens, tags))
    return sentences


def group_token_rows(rows: list[tuple[str, str]]) -> list[tuple[list[str], list[list[str]]]]:
    """Group (word, tag) rows that carry no sentence marker into sentences,
    cutting after sentence-final punctuation."""
    sentences: list[tuple[list[str], list[list[str]]]] = []
    tokens: list[str] = []
    tags: list[list[str]] = []
    for word, tag in rows:
        if not word:
            if tokens:
                sentences.append((tokens, tags))
                tokens, tags = [], []
            continue
        tokens.append(word)
        tags.append([tag or "O"])
        if word in _SENTENCE_FINAL:
            sentences.append((tokens, tags))
            tokens, tags = [], []
    if tokens:
        sentences.append((tokens, tags))
    return sentences


def load_wojood(paths: list[str]) -> list[dict]:
    examples = []
    for path in paths:
        for tokens, tags in read_conll_sentences(path):
            examples.append(tagged_sentence_to_example(tokens, tags, WOJOOD_TYPE_MAP, "wojood"))
    return examples


def load_anercorp_file(path: str) -> list[dict]:
    """Local two-column file: `word<space or tab>tag` per line."""
    rows: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            rows.append((parts[0], parts[-1]) if len(parts) >= 2 else ("", ""))
    return [tagged_sentence_to_example(t, g, ANERCORP_TYPE_MAP, "anercorp") for t, g in group_token_rows(rows)]


def load_anercorp_hf(name: str, splits: list[str]) -> list[dict]:
    """Download from the Hugging Face Hub (needs `datasets` and internet)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("--anercorp-hf needs the 'datasets' package: pip install -r requirements-train.txt") from None
    dataset = load_dataset(name)
    rows: list[tuple[str, str]] = []
    for split in splits:
        if split not in dataset:
            raise SystemExit(f"split '{split}' not found in {name}; available: {list(dataset.keys())}")
        columns = dataset[split].column_names
        if "word" not in columns or "tag" not in columns:
            raise SystemExit(f"expected columns 'word' and 'tag' in {name}, found {columns}")
        rows.extend(zip(dataset[split]["word"], dataset[split]["tag"]))
        rows.append(("", ""))
    return [tagged_sentence_to_example(t, g, ANERCORP_TYPE_MAP, "anercorp") for t, g in group_token_rows(rows)]


def load_team_jsonl(paths: list[str]) -> list[dict]:
    """Team-written sentences in the golden-test row format. Only the types the
    model is trained to produce are kept as targets; identifiers are left to regex."""
    examples = []
    for path in paths:
        for row in read_jsonl(path):
            try:
                spans = gold_spans(row)
            except GoldFormatError as exc:
                raise SystemExit(f"{path}: {exc}") from None
            text = row["text"]
            entities = [{"text": text[s.start:s.end], "type": s.type} for s in spans if s.type in MODEL_TYPES]
            examples.append({"text": text, "entities": entities, "source": "team"})
    return examples


# --------------------------------------------------------------------------- #
# Template-generated synthetic sentences (all names invented)
# --------------------------------------------------------------------------- #
_SYN_FIRST = (
    "عبدالعزيز سلمان بندر مشعل نواف ثامر راشد وليد حمد زياد يزيد عمر إبراهيم يوسف صالح "
    "هيفاء أمل دانة شهد رهف غادة لطيفة العنود بشاير وعد أسماء حصة مها رزان خلود"
).split()
_SYN_FAMILY = (
    "العمري الجهني الرشيدي المالكي الحازمي الثبيتي الخالدي اليامي العسيري الفيفي "
    "البقمي الحمدان النعيمي السلمي الصاعدي القرني الشهراني العلياني الرويلي المحمدي"
).split()
_SYN_CITIES = ["الرياض", "جدة", "الدمام", "أبها", "تبوك", "حائل", "الطائف", "نجران", "جازان", "ينبع", "عرعر", "سكاكا", "الخرج", "القطيف", "الباحة", "بيشة"]
_SYN_DISTRICTS = ["حي الورود", "حي الفيصلية", "حي المروج", "حي السلامة", "حي الشاطئ", "حي العزيزية", "حي الخالدية", "شارع الأمير", "طريق الملك"]
_SYN_ORGS = [
    "مستشفى الصفا العام", "مركز الريان الطبي", "مجمع الندى الطبي", "مستشفى الأمل التخصصي", "عيادات الشروق",
    "مركز البيان لطب الأسنان", "مستشفى الهدى", "شركة الدرع للتأمين الطبي", "مركز الرعاية الأولية بالنسيم", "مختبرات الدقة الطبية",
]
_SYN_CLINICS = ["الباطنة", "العظام", "الأسنان", "القلب", "الجلدية", "العيون", "الأطفال", "الطوارئ", "السكري", "الأنف والأذن والحنجرة"]
_G_MONTHS = "يناير فبراير مارس أبريل مايو يونيو يوليو أغسطس سبتمبر أكتوبر نوفمبر ديسمبر".split()
_H_MONTHS = ["محرم", "صفر", "ربيع الأول", "ربيع الآخر", "جمادى الأولى", "جمادى الآخرة", "رجب", "شعبان", "رمضان", "شوال", "ذو القعدة", "ذو الحجة"]

# Fields in braces: pers, pers2, loc, loc2, org, date are model targets;
# clinic, nid, phone, mrn are fillers the model must leave alone.
_SYN_TEMPLATES = [
    "راجع المريض {pers} عيادة {clinic} في {org} بتاريخ {date} لمتابعة حالته.",
    "تم تنويم المريضة {pers} في {org} يوم {date} بعد تحويلها من {loc}.",
    "أفاد الدكتور {pers} بأن المريض {pers2} يحتاج إلى مراجعة عيادة {clinic} خلال أسبوعين.",
    "المريض {pers}، رقم الهوية {nid}، يسكن في {loc} وجواله {phone}.",
    "رقم الملف الطبي {mrn} يعود للمريضة {pers} التي راجعت {org} في {date}.",
    "صدر التقرير الطبي من {org} في {loc} بتاريخ {date}.",
    "طلبت الممرضة {pers} تحديث بيانات المراجع {pers2} في نظام {org}.",
    "حُوّلت الحالة إلى {org} بـ{loc} لعدم توفر سرير في عيادة {clinic}.",
    "موعد العملية للمريض {pers} في {date}، والتواصل مع قريبه {pers2} على {phone}.",
    "يقيم المريض في {loc}، {loc2}، ويراجع {org} منذ {date}.",
    "اعتذر الدكتور {pers} عن عيادة {clinic} يوم {date} وسيغطي مكانه الدكتور {pers2}.",
    "وصلت المريضة {pers} إلى طوارئ {org} قادمة من {loc} وهي تشكو من ضيق في التنفس.",
    "تمت الموافقة على طلب التأمين المقدم من {org} للمريض {pers} بتاريخ {date}.",
    "خرج المريض {pers} من {org} في {date} مع توصية بالراحة.",
    # Saudi dialect
    "{pers} راح {org} أمس وقالوا له موعدك {date}.",
    "والدة {pers} تقول إن بنتها {pers2} تعبانة وودّوها عيادة {clinic} في {loc}.",
    "كلمت الدكتور {pers} وقال لازم تجون {org} قبل {date}.",
    "ولد عمي {pers} ساكن في {loc} ويبي يحوّل ملفه على {org}.",
    "يا أخي ترى {pers} جواله {phone} ورقم ملفه {mrn}، كلموه إذا طلعت النتيجة.",
    # No sensitive information
    "يُنصح المريض بتناول الدواء بعد الأكل مرتين يومياً لمدة عشرة أيام.",
    "أظهرت التحاليل ارتفاعاً بسيطاً في السكر التراكمي ويُعاد الفحص بعد ثلاثة أشهر.",
    "عيادة {clinic} تستقبل المراجعين من الثامنة صباحاً حتى الرابعة عصراً.",
    "ضغط الدم 130 على 85 والنبض 78 والحرارة 37 درجة.",
    "يجب الصيام ثماني ساعات قبل تحليل الدهون.",
    "لا تنسون تجيبون التقارير القديمة معكم وقت الموعد.",
    "تم صرف مضاد حيوي ومسكن للألم مع تعليمات العناية بالجرح.",
]
_TARGET_FIELDS = {"pers": "PERS", "pers2": "PERS", "loc": "LOC", "loc2": "LOC", "org": "ORG", "date": "DATE"}


def _syn_person(rng: random.Random) -> str:
    first = rng.choice(_SYN_FIRST)
    return first if rng.random() < 0.3 else f"{first} {rng.choice(_SYN_FAMILY)}"


def _syn_date(rng: random.Random) -> str:
    day, kind = rng.randint(1, 28), rng.random()
    if kind < 0.35:
        return f"{day} {rng.choice(_G_MONTHS)} {rng.randint(2019, 2026)}"
    if kind < 0.55:
        return f"{day} {rng.choice(_H_MONTHS)} {rng.randint(1441, 1448)}هـ"
    if kind < 0.70:
        return f"{rng.choice(_G_MONTHS)} {rng.randint(2019, 2026)}"
    month, year = rng.randint(1, 12), rng.randint(2019, 2026)
    return rng.choice([f"{day:02d}/{month:02d}/{year}", f"{year}-{month:02d}-{day:02d}", f"{day}-{month}-{year}"])


def generate_synthetic(n: int, seed: int) -> list[dict]:
    """Seeded, reproducible template sentences. They teach the clinical register
    and that identifiers (ID, phone, MRN) are not the model's job."""
    rng = random.Random(seed)
    examples: list[dict] = []
    for i in range(n):
        template = _SYN_TEMPLATES[i % len(_SYN_TEMPLATES)]
        values = {
            "pers": _syn_person(rng),
            "pers2": _syn_person(rng),
            "loc": rng.choice(_SYN_CITIES),
            "loc2": rng.choice(_SYN_DISTRICTS),
            "org": rng.choice(_SYN_ORGS),
            "date": _syn_date(rng),
            "clinic": rng.choice(_SYN_CLINICS),
            "nid": f"{rng.choice('12')}{rng.randint(0, 999999999):09d}",
            "phone": f"05{rng.randint(0, 99999999):08d}",
            "mrn": f"{rng.randint(10000, 9999999)}",
        }
        while values["pers2"] == values["pers"]:
            values["pers2"] = _syn_person(rng)
        text = template.format(**values)
        targets = [(template.index("{" + f + "}"), values[f], t) for f, t in _TARGET_FIELDS.items() if "{" + f + "}" in template]
        entities = [{"text": v, "type": t} for _, v, t in sorted(targets)]
        examples.append({"text": text, "entities": entities, "source": "template"})
    return examples


# --------------------------------------------------------------------------- #
# Overlap with the golden test
# --------------------------------------------------------------------------- #
MIN_CONTAINMENT_CHARS = 10


class GoldenIndex:
    """Decides whether a training sentence collides with a golden sentence.

    exact           equal after normalisation
    contained       one is a substring of the other
    near_duplicate  Jaccard similarity of character 5-grams >= `threshold`
    shared_content  the shorter sentence shares >= `shared_threshold` of its
                    5-grams with the longer one. This catches a golden sentence
                    with a word changed and a clause added, which lowers the
                    Jaccard score but leaves most of the sentence intact.

    Known limit: a sentence that keeps only the frame of a golden sentence and
    replaces its names and dates can share less than the threshold and pass.
    """

    def __init__(self, golden_rows: list[dict], threshold: float, shared_threshold: float = 0.6) -> None:
        self.threshold = threshold
        self.shared_threshold = shared_threshold
        self.items = []
        for row in golden_rows:
            norm = normalize_for_dedup(row["text"])
            self.items.append((row.get("id", "?"), norm, char_shingles(norm)))
        self.exact = {norm: gid for gid, norm, _ in self.items}

    def match(self, text: str) -> dict | None:
        """Return why `text` collides with the golden test, or None."""
        norm = normalize_for_dedup(text)
        if norm in self.exact:
            return {"golden_id": self.exact[norm], "kind": "exact", "score": 1.0}
        shingles = char_shingles(norm)
        for gid, g_norm, g_shingles in self.items:
            if min(len(g_norm), len(norm)) >= MIN_CONTAINMENT_CHARS and (g_norm in norm or norm in g_norm):
                return {"golden_id": gid, "kind": "contained", "score": 1.0}
            if not shingles or not g_shingles:
                continue
            shared = len(shingles & g_shingles)
            score = shared / (len(shingles) + len(g_shingles) - shared)
            if score >= self.threshold:
                return {"golden_id": gid, "kind": "near_duplicate", "score": round(score, 3)}
            coverage = shared / min(len(shingles), len(g_shingles))
            if coverage >= self.shared_threshold:
                return {"golden_id": gid, "kind": "shared_content", "score": round(coverage, 3)}
        return None


def remove_golden_overlap(examples: list[dict], index: GoldenIndex) -> tuple[list[dict], list[dict]]:
    kept, removed = [], []
    for ex in examples:
        hit = index.match(ex["text"])
        if hit:
            removed.append({"source": ex["source"], "text": ex["text"], **hit})
        else:
            kept.append(ex)
    return kept, removed


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def clean_examples(examples: list[dict]) -> list[dict]:
    """Drop fragments, overlong sentences and duplicates (first occurrence wins)."""
    seen: set[str] = set()
    out = []
    for ex in examples:
        text = ex["text"].strip()
        if len(text.split()) < MIN_TOKENS or len(text) > MAX_CHARS:
            continue
        norm = normalize_for_dedup(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append({**ex, "text": text})
    return out


def cap_negatives(examples: list[dict], max_frac: float, rng: random.Random) -> list[dict]:
    """Keep sentences without entities to at most `max_frac` of the set."""
    positives = [e for e in examples if e["entities"]]
    negatives = [e for e in examples if not e["entities"]]
    if max_frac >= 1 or not positives:
        return examples
    limit = int(len(positives) * max_frac / (1 - max_frac))
    if len(negatives) > limit:
        negatives = rng.sample(negatives, limit)
    return positives + negatives


def prepare(args: argparse.Namespace) -> dict:
    golden_path = Path(args.golden)
    if not golden_path.exists():
        raise SystemExit(
            f"{golden_path} not found. The golden test must be written and committed before any training data is built."
        )
    golden_rows = read_jsonl(golden_path)
    errors, _, _ = validate_rows(golden_rows, min_rows=args.min_golden_rows, require_review=not args.allow_unreviewed)
    if errors:
        raise SystemExit("golden test is not valid; run `python -m src.validate_golden` and fix:\n  " + "\n  ".join(errors[:10]))
    index = GoldenIndex(golden_rows, args.near_dup_threshold, args.shared_content_threshold)

    for shot_text, _ in FEW_SHOT_EXAMPLES:
        hit = index.match(shot_text)
        if hit:
            raise SystemExit(f"a few-shot prompt example collides with golden row {hit['golden_id']}; change it in src/prompts.py")

    examples: list[dict] = []
    if args.wojood:
        examples += load_wojood(args.wojood)
    if args.anercorp_file:
        examples += load_anercorp_file(args.anercorp_file)
    if args.anercorp_hf:
        examples += load_anercorp_hf(args.anercorp_hf, args.anercorp_splits)
    if args.team_jsonl:
        examples += load_team_jsonl(args.team_jsonl)
    if args.synthetic_n > 0:
        examples += generate_synthetic(args.synthetic_n, args.seed)
    if not examples:
        raise SystemExit("no training source given; pass --wojood, --anercorp-file, --anercorp-hf, --team-jsonl or --synthetic-n")

    loaded = Counter(e["source"] for e in examples)
    vague_dates_dropped = sum(e.get("vague_dates_dropped", 0) for e in examples)
    examples = clean_examples(examples)
    examples, removed = remove_golden_overlap(examples, index)

    rng = random.Random(args.seed)
    examples = cap_negatives(examples, args.max_neg_frac, rng)
    rng.shuffle(examples)

    # Verify the guarantee on the final set rather than trusting the filter above.
    remaining = [index.match(e["text"]) for e in examples]
    remaining = [r for r in remaining if r]
    if remaining:
        raise SystemExit(f"overlap check failed: {len(remaining)} training example(s) still collide with the golden test")

    n_val = max(1, int(len(examples) * args.val_frac)) if len(examples) > 1 else 0
    val, train = examples[:n_val], examples[n_val:]

    def to_row(ex: dict) -> dict:
        return {**build_training_example(ex["text"], ex["entities"]), "source": ex["source"]}

    out = Path(args.out)
    write_jsonl(out / "train.jsonl", (to_row(e) for e in train))
    write_jsonl(out / "val.jsonl", (to_row(e) for e in val))

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "golden_file": str(golden_path),
        "golden_sha256": sha256_file(golden_path),
        "golden_rows": len(golden_rows),
        "near_dup_threshold": args.near_dup_threshold,
        "shared_content_threshold": args.shared_content_threshold,
        "seed": args.seed,
        "loaded_by_source": dict(loaded),
        "vague_date_labels_dropped": vague_dates_dropped,
        "removed_for_golden_overlap": len(removed),
        "removed_examples": removed,
        "overlap_after_filtering": 0,
        "train_rows": len(train),
        "val_rows": len(val),
        "final_by_source": dict(Counter(e["source"] for e in examples)),
        "entities_by_type": dict(Counter(ent["type"] for e in examples for ent in e["entities"])),
        "rows_without_entities": sum(1 for e in examples if not e["entities"]),
    }
    (out / "prep_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build instruction-tuning JSONL and verify zero overlap with the golden test.")
    p.add_argument("--golden", default="data/golden_test.jsonl")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--wojood", nargs="*", default=[], help="Wojood CoNLL files")
    p.add_argument("--anercorp-file", help="local two-column ANERCorp file")
    p.add_argument("--anercorp-hf", help="Hugging Face dataset name, e.g. asas-ai/ANERCorp")
    p.add_argument("--anercorp-splits", nargs="+", default=["train"])
    p.add_argument("--team-jsonl", nargs="*", default=[], help="team-written sentences in the golden row format")
    p.add_argument("--synthetic-n", type=int, default=600, help="number of template sentences (0 to disable)")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--near-dup-threshold", type=float, default=0.8)
    p.add_argument("--shared-content-threshold", type=float, default=0.6,
                   help="share of the shorter sentence's character 5-grams found in the other one")
    p.add_argument("--max-neg-frac", type=float, default=0.25, help="largest share of sentences without entities")
    p.add_argument("--min-golden-rows", type=int, default=100)
    p.add_argument("--allow-unreviewed", action="store_true", help="accept a golden file whose rows are not yet double-checked")
    return p


def main(argv: list[str] | None = None) -> int:
    report = prepare(build_parser().parse_args(argv))
    print(f"golden test : {report['golden_rows']} rows, sha256 {report['golden_sha256'][:16]}…")
    print(f"loaded      : {report['loaded_by_source']}")
    print(f"removed     : {report['removed_for_golden_overlap']} (overlap with golden test)")
    print(f"final       : {report['train_rows']} train / {report['val_rows']} val  by source {report['final_by_source']}")
    print(f"entities    : {report['entities_by_type']}  | rows without entities: {report['rows_without_entities']}")
    print("overlap with golden test after filtering: 0  (verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
