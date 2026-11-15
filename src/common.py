"""Shared types and text utilities for the Arabic PII de-identification project.

Everything here is pure standard library so it runs anywhere (laptop, Colab, CI).
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------------- #
# Entity types
# --------------------------------------------------------------------------- #
MODEL_TYPES = ("PERS", "LOC", "ORG", "DATE")          # produced by the LLM
REGEX_TYPES = ("NATIONAL_ID", "PHONE", "MRN", "EMAIL")  # produced by regex tools only
ALL_TYPES = MODEL_TYPES + REGEX_TYPES                  # DATE comes from both layers

PLACEHOLDERS = {
    "PERS": "[اسم]",
    "LOC": "[موقع]",
    "ORG": "[جهة]",
    "DATE": "[تاريخ]",
    "NATIONAL_ID": "[هوية]",
    "PHONE": "[جوال]",
    "MRN": "[ملف طبي]",
    "EMAIL": "[بريد]",
}

# Label spellings a model commonly produces; mapped so a near-miss label is not
# silently discarded. Anything else is dropped and counted.
TYPE_ALIASES = {
    "PER": "PERS",
    "PERSON": "PERS",
    "GPE": "LOC",
    "FAC": "LOC",
    "LOCATION": "LOC",
    "ORGANIZATION": "ORG",
}


@dataclass(frozen=True, order=True)
class Span:
    """A half-open character span [start, end) in the original text."""

    start: int
    end: int
    type: str
    source: str = "model"  # "model" | "regex" | "gold" | "repair"

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end

    def to_dict(self, text: str | None = None) -> dict:
        d = {"start": self.start, "end": self.end, "type": self.type, "source": self.source}
        if text is not None:
            d["text"] = text[self.start:self.end]
        return d


# --------------------------------------------------------------------------- #
# Normalisation (used for duplicate / overlap detection, never for display)
# --------------------------------------------------------------------------- #
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
ZERO_WIDTH = "​‌‍‎‏⁠﻿؜"
_ZERO_WIDTH_RE = re.compile(f"[{ZERO_WIDTH}]")
DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})
_LETTER_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
_NON_ALNUM = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_for_dedup(text: str) -> str:
    """Aggressive normalisation so trivially edited copies compare equal."""
    t = unicodedata.normalize("NFKC", text)
    t = _ZERO_WIDTH_RE.sub("", t).replace(TATWEEL, "")
    t = _DIACRITICS.sub("", t)
    t = t.translate(DIGIT_MAP).translate(_LETTER_MAP)
    t = _NON_ALNUM.sub(" ", t).replace("_", " ")
    return _WS.sub(" ", t).strip().lower()


def char_shingles(norm_text: str, n: int = 5) -> frozenset[str]:
    if len(norm_text) <= n:
        return frozenset({norm_text}) if norm_text else frozenset()
    return frozenset(norm_text[i:i + n] for i in range(len(norm_text) - n + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


# --------------------------------------------------------------------------- #
# Arabic-aware occurrence search
# --------------------------------------------------------------------------- #
# Single-letter particles written attached to the next word. "ك" is left out on
# purpose: it rarely precedes a proper name, and accepting it would cut "ريم"
# out of "كريم".
PROCLITICS = "وفبل"

# Characters that carry no letter of their own: zero-width characters, tatweel and
# Arabic diacritics. A name must still be found when any of them sits inside it.
_REMOVABLE = set(ZERO_WIDTH) | {TATWEEL} | {
    chr(c) for lo, hi in ((0x610, 0x61A), (0x64B, 0x65F), (0x670, 0x670), (0x6D6, 0x6ED)) for c in range(lo, hi + 1)
}
_MATCH_LETTERS = {ord(k): v for k, v in {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"}.items()}


def match_view(text: str) -> tuple[str, list[int]]:
    """A normalised copy of `text` used only for locating strings, with a map
    from every character of the copy to its index in `text`.

    Diacritics, tatweel and zero-width characters are dropped, runs of
    whitespace (including newlines) become one space, digits are mapped to
    ASCII and common letter variants are unified.
    """
    chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        if ch in _REMOVABLE:
            continue
        if ch.isspace():
            if chars and chars[-1] == " ":
                continue
            ch = " "
        else:
            ch = DIGIT_MAP.get(ord(ch)) or _MATCH_LETTERS.get(ord(ch)) or ch
        chars.append(ch)
        index_map.append(i)
    return "".join(chars), index_map


def _original_span(text: str, index_map: list[int], i: int, j: int) -> tuple[int, int]:
    start, end = index_map[i], index_map[j - 1] + 1
    while end < len(text) and text[end] in _REMOVABLE:
        end += 1  # keep a trailing diacritic or tatweel with the word it belongs to
    return start, end


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _left_strict(view: str, i: int, needle: str) -> bool:
    return not _is_word_char(needle[0]) or i == 0 or not _is_word_char(view[i - 1])


def _left_proclitic(view: str, i: int) -> bool:
    """One or two attached proclitics before the match, e.g. "لخالد", "وبمحمد"."""
    k = i
    for _ in range(2):
        if k > 0 and view[k - 1] in PROCLITICS:
            k -= 1
            if k == 0 or not _is_word_char(view[k - 1]):
                return True
        else:
            break
    return False


def _right_ok(view: str, j: int, needle: str) -> bool:
    return not _is_word_char(needle[-1]) or j >= len(view) or not _is_word_char(view[j])


# Words that are often typed glued to the name that follows them ("أبومحمد"),
# in the spelling `match_view` produces.
KUNYA_PREFIXES = ("ابو", "ابي", "ابا", "ام", "ابن", "بن", "بنت", "عبد")
FATHATAN = "\u064b"


def _left_kunya(view: str, i: int) -> bool:
    for prefix in KUNYA_PREFIXES:
        k = i - len(prefix)
        if k >= 0 and view.startswith(prefix, k) and (k == 0 or not _is_word_char(view[k - 1]) or _left_proclitic(view, k)):
            return True
    return False


def _right_case_ending(view: str, j: int, text: str, index_map: list[int]) -> bool:
    """The accusative ending of an indefinite noun: an alef carrying tanween
    ("محمدًا"). Only accepted when the tanween mark is really written."""
    if j >= len(view) or view[j] != "ا" or (j + 1 < len(view) and _is_word_char(view[j + 1])):
        return False
    last, alef = index_map[j - 1], index_map[j]
    after = index_map[j + 1] if j + 1 < len(index_map) else len(text)
    return FATHATAN in text[last + 1:alef] or FATHATAN in text[alef + 1:after]


def _scan(view: str, needle: str, mode: str, text: str = "", index_map: list[int] | None = None) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    pos = 0
    while True:
        i = view.find(needle, pos)
        if i < 0:
            return out
        j = i + len(needle)
        if mode == "plain":
            ok = True
        else:
            left = _left_strict(view, i, needle)
            right = _right_ok(view, j, needle)
            if mode in ("loose", "attached"):
                left = left or _left_proclitic(view, i)
            if mode == "attached":
                left = left or _left_kunya(view, i)
                right = right or _right_case_ending(view, j, text, index_map or [])
            ok = left and right
        if ok:
            out.append((i, j))
            pos = j
        else:
            pos = i + 1


def find_occurrences(text: str, needle: str, mode: str = "auto") -> list[tuple[int, int]]:
    """Occurrences of `needle` in `text`, as spans of the original text.

    strict    word-bounded on both sides, so "علي" is not found inside "عليه".
    loose     also accepts an attached proclitic before the match ("لمحمد").
    attached  loose, plus a glued kunya word before it ("أبومحمد") and a
              written case ending after it ("محمدًا"). A name buried in an
              unrelated longer word ("الرياض" in "الرياضة") is never matched.
    auto      strict matches; attached ones only when there is no strict match,
              so that "ريم" is not cut out of "وريم" when "ريم" also stands alone.
    plain     any substring.
    """
    view, index_map = match_view(text)
    target = match_view(needle)[0].strip()
    if not target:
        return []
    if mode == "auto":
        hits = _scan(view, target, "strict") or _scan(view, target, "attached", text, index_map)
    else:
        hits = _scan(view, target, mode, text, index_map)
    return [_original_span(text, index_map, i, j) for i, j in hits]


def find_plain(text: str, needle: str) -> list[tuple[int, int]]:
    """All non-overlapping substring occurrences, with no word-boundary rule."""
    return find_occurrences(text, needle, mode="plain")


# --------------------------------------------------------------------------- #
# Model output parsing
# --------------------------------------------------------------------------- #
class ModelOutputError(ValueError):
    """The model reply could not be turned into a valid entity list."""


@dataclass
class ParsedOutput:
    entities: list[dict]   # [{"text": str, "type": str}]
    cleaned: bool          # True if a fence / think block / stray text had to be removed
    dropped_items: int     # items with a missing text or an unknown type


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_FENCE_OPEN = re.compile(r"^```[A-Za-z]*\s*")
_FENCE_CLOSE = re.compile(r"\s*```$")


def parse_model_output(raw: str) -> ParsedOutput:
    """Parse `{"entities": [{"text": ..., "type": ...}]}` from a model reply.

    Chat models often wrap JSON in a markdown fence; that wrapper is removed and
    reported through `cleaned` rather than counted as a failure.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ModelOutputError("empty model output")
    original = raw.strip()
    s = _THINK_RE.sub("", original).strip()
    if s.startswith("```"):
        s = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", s)).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find("{"), s.rfind("}")
        if a < 0 or b <= a:
            raise ModelOutputError("no JSON object in model output") from None
        try:
            obj = json.loads(s[a:b + 1])
        except json.JSONDecodeError as exc:
            raise ModelOutputError(f"invalid JSON: {exc.msg}") from None
        s = s[a:b + 1]
    cleaned = s != original
    if not isinstance(obj, dict) or not isinstance(obj.get("entities"), list):
        raise ModelOutputError("JSON does not contain an 'entities' list")

    entities: list[dict] = []
    seen: set[tuple[str, str]] = set()
    dropped = 0
    for item in obj["entities"]:
        if not isinstance(item, dict):
            dropped += 1
            continue
        text, typ = item.get("text"), item.get("type")
        if not isinstance(text, str) or not text.strip() or not isinstance(typ, str):
            dropped += 1
            continue
        typ = typ.strip().upper()
        typ = TYPE_ALIASES.get(typ, typ)
        if typ not in MODEL_TYPES:
            dropped += 1
            continue
        key = (text.strip(), typ)
        if key not in seen:
            seen.add(key)
            entities.append({"text": key[0], "type": typ})
    return ParsedOutput(entities=entities, cleaned=cleaned, dropped_items=dropped)


# --------------------------------------------------------------------------- #
# Spans: building, overlap resolution, redaction
# --------------------------------------------------------------------------- #
def spans_from_entities(text: str, entities: Iterable[dict], source: str = "model") -> tuple[list[Span], list[dict]]:
    """Locate every entity string in `text`. Returns (spans, entities_not_found_in_text)."""
    spans: set[Span] = set()
    missing: list[dict] = []
    for ent in entities:
        occ = find_occurrences(text, ent["text"])
        if not occ:
            missing.append(ent)
        for a, b in occ:
            spans.add(Span(a, b, ent["type"], source))
    return sorted(spans), missing


def _strength(s: Span) -> tuple[int, int, int]:
    # Deterministic tools outrank the model; then the longer span; then the earlier one.
    return (1 if s.source == "regex" else 0, s.length, -s.start)


def resolve_overlaps(spans: Iterable[Span]) -> list[Span]:
    """Make spans disjoint.

    Nested span  -> the longer (outer) span is kept.
    Partial overlap -> the two are merged into one span covering both, so no
    character that either layer flagged is left unmasked; the merged span takes
    the type of the stronger one.
    """
    ordered = sorted(spans, key=lambda s: (s.start, -s.length, s.type, s.source))
    out: list[Span] = []
    for s in ordered:
        if out and s.start < out[-1].end:
            last = out[-1]
            if s.end <= last.end:  # nested (or identical extent)
                if (s.start, s.end) == (last.start, last.end) and _strength(s) > _strength(last):
                    out[-1] = s
                continue
            winner = s if _strength(s) > _strength(last) else last
            out[-1] = Span(last.start, s.end, winner.type, winner.source)
        else:
            out.append(s)
    return out


def redact(text: str, spans: Iterable[Span]) -> str:
    """Replace each (disjoint) span with the placeholder of its type."""
    out: list[str] = []
    pos = 0
    for s in resolve_overlaps(spans):
        out.append(text[pos:s.start])
        out.append(PLACEHOLDERS.get(s.type, "[محجوب]"))
        pos = s.end
    out.append(text[pos:])
    return "".join(out)


def coverage_mask(length: int, spans: Iterable[Span]) -> bytearray:
    mask = bytearray(length)
    for s in spans:
        for i in range(max(0, s.start), min(length, s.end)):
            mask[i] = 1
    return mask


def is_fully_covered(span: Span, mask: bytearray) -> bool:
    return all(mask[i] for i in range(span.start, span.end))


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {n} is not valid JSON ({exc.msg})") from None
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Gold annotations
# --------------------------------------------------------------------------- #
class GoldFormatError(ValueError):
    pass


def gold_spans(example: dict) -> list[Span]:
    """Turn the `entities` of a golden-test row into spans.

    Each mention is listed once, in reading order. `start` is optional: when it
    is absent the next unused occurrence of the entity text is taken, which lets
    annotators write rows by hand without counting characters.
    """
    text = example["text"]
    rid = example.get("id")
    entities = example.get("entities", [])
    for ent in entities:
        etext, etype = ent.get("text"), ent.get("type")
        if not isinstance(etext, str) or not etext:
            raise GoldFormatError(f"{rid}: entity without text")
        if etype not in ALL_TYPES:
            raise GoldFormatError(f"{rid}: unknown type {etype!r}")

    # Positions for mentions written without `start`, decided per entity string.
    mentions: dict[str, int] = {}
    for ent in entities:
        if "start" not in ent:
            mentions[ent["text"]] = mentions.get(ent["text"], 0) + 1
    positions: dict[str, list[int]] = {}
    for etext, count in mentions.items():
        if any("start" in e for e in entities if e["text"] == etext):
            raise GoldFormatError(f"{rid}: give 'start' for every mention of {etext!r} or for none")
        plain: list[int] = []
        pos = text.find(etext)
        while pos >= 0:
            plain.append(pos)
            pos = text.find(etext, pos + len(etext))
        if not plain:
            raise GoldFormatError(f"{rid}: entity text {etext!r} not found in text")
        strict = {a for a, _ in find_occurrences(text, etext, "strict")}
        attached = {a for a, _ in find_occurrences(text, etext, "attached")}
        # Every listed mention must sit on a word of its own (a particle or kunya
        # word may be attached). This keeps "علي" from being placed inside "عليه".
        # Anything else needs an explicit `start`, so nothing is placed by guesswork.
        options = [[p for p in plain if p in strict], [p for p in plain if p in attached]]
        if all(p in attached for p in plain):
            options.insert(0, plain)
        for candidates in options:
            if len(candidates) == count:
                positions[etext] = candidates
                break
        else:
            raise GoldFormatError(
                f"{rid}: cannot tell which occurrence(s) of {etext!r} are meant "
                f"(listed {count} time(s), found {len(plain)} time(s), {len(attached)} as a separate word); "
                "add a 'start' offset to each mention"
            )

    spans: list[Span] = []
    used: dict[str, int] = {}
    for ent in entities:
        etext, etype = ent["text"], ent["type"]
        if "start" in ent:
            a = ent["start"]
            b = ent.get("end", a + len(etext))
            if not isinstance(a, int) or text[a:b] != etext:
                raise GoldFormatError(f"{rid}: offsets do not match entity text {etext!r}")
        else:
            a = positions[etext][used.get(etext, 0)]
            b = a + len(etext)
            used[etext] = used.get(etext, 0) + 1
        # A diacritic or tatweel written right after the mention belongs to it, as
        # it does for predicted spans; otherwise an exact prediction would be
        # scored as a boundary error.
        while b < len(text) and text[b] in _REMOVABLE:
            b += 1
        spans.append(Span(a, b, etype, "gold"))
    spans.sort()
    for prev, cur in zip(spans, spans[1:]):
        if cur.start < prev.end:
            raise GoldFormatError(
                f"{example.get('id')}: gold entities overlap ({text[prev.start:prev.end]!r} / {text[cur.start:cur.end]!r}); "
                "annotate the outermost entity only"
            )
    return spans
