"""Deterministic tools for identifiers that follow a fixed pattern.

These run independently of the language model, so a model failure or a prompt
injection cannot switch them off.

Matching is done on a normalised copy of the text (Arabic-Indic digits mapped
to ASCII, zero-width characters and tatweel removed) and the spans are mapped
back to the original text, so obfuscated digits are still found and the
returned offsets always refer to the text the caller passed in.
"""
from __future__ import annotations

import re

from .common import DIGIT_MAP, TATWEEL, ZERO_WIDTH, Span

_REMOVE = set(ZERO_WIDTH) | {TATWEEL}


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Return (normalised_text, index_map) where index_map[k] is the position in
    `text` of normalised character k."""
    chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        if ch in _REMOVE:
            continue
        chars.append(DIGIT_MAP.get(ord(ch), ch))
        index_map.append(i)
    return "".join(chars), index_map


def _to_original(text: str, index_map: list[int], start: int, end: int) -> tuple[int, int]:
    """Map a span of the normalised text back to the original. Characters that
    normalisation removed right after the match (e.g. the tatweel of "هـ") are
    included so they do not dangle after the placeholder."""
    o_start, o_end = index_map[start], index_map[end - 1] + 1
    while o_end < len(text) and text[o_end] in _REMOVE:
        o_end += 1
    return o_start, o_end


# Up to three separator characters are tolerated between digits
# ("0 5 5 ...", "05-5-...", "055 - 123 - 4567").
_SEP = r"[ \-.]{0,3}"

_PHONE_PATTERNS = [
    # International form: +966 5x xxx xxxx, 00966..., 966..., +966 (0) 5x ...
    re.compile(rf"(?<![\d+])(?:(?:\+|00)[ ]?966|966){_SEP}\(?0?\)?{_SEP}5(?:{_SEP}\d){{8}}(?!\d)"),
    # National form with the prefix in parentheses: (055) 123 4567
    re.compile(rf"(?<!\d)\(05\d\)(?:{_SEP}\d){{7}}(?!\d)"),
    # National form: 05x xxx xxxx
    re.compile(rf"(?<!\d)05(?:{_SEP}\d){{8}}(?!\d)"),
]

# Ten digits starting with 1 or 2. A colon or slash followed by a digit right
# after the match means a time or a date ("... 10:30"), so that is rejected.
_ID_PATTERN = re.compile(r"(?<!\d)[12](?:[ \-]{0,2}\d){9}(?!\d)(?![:/]\d)")
_ID_SEPARATOR = re.compile(r"[ \-]+")
_ID_CUES = ("هوي", "إقام", "اقام", "مدني", "id", "iqama")
_MAX_ID_GROUPS = 5      # "1012-345-678" has 3 groups; a list "1 2 3 ..." has ten
_ID_CUE_WINDOW = 25

# A cue that contains "رقم" or "MRN" is specific enough on its own. The bare
# phrases "الملف الطبي" / "السجل الطبي" need a colon or "#" before the number,
# otherwise "حُدّث الملف الطبي 2023" would lose its year.
_MRN_NUMBER = r"([A-Za-z]{0,3}-?\d{4,12})(?!\d)"
_MRN_PATTERNS = [
    re.compile(
        r"(?:MRN|M\.R\.N\.?"
        r"|رقم\s+(?:ال)?ملف\w{0,3}(?:\s+(?:ال)?(?:طبي|مريضة|مريض))?"
        r"|(?:ال)?ملف\w{0,3}\s+(?:(?:ال)?طبي\s+)?رقم\w{0,2}"
        r"|رقم\s+السجل(?:\s+الطبي)?"
        r"|السجل\s+الطبي\s+رقم)"
        r"(?:\s+(?:هو|هي))?\s*[:#\-]?\s*" + _MRN_NUMBER,
        re.IGNORECASE,
    ),
    re.compile(r"(?:الملف\s+الطبي|السجل\s+الطبي)\s*[:#]\s*" + _MRN_NUMBER),
]

_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")

# day-month-year or year-month-day with one separator used throughout.
# A two-digit year is accepted only with "/", so version numbers and dose
# ranges ("1.2.20", "5-10-15") are not read as dates.
_DATE_PATTERN = re.compile(
    r"(?<![\d/.\-])"
    r"(?:(?P<d1>\d{1,2})(?P<s1>[/\-.])(?P<m1>\d{1,2})(?P=s1)(?P<y1>\d{4}|\d{2})"
    r"|(?P<y2>\d{4})(?P<s2>[/\-.])(?P<m2>\d{1,2})(?P=s2)(?P<d2>\d{1,2}))"
    r"(?![\d/\-])"
    r"(?:\s?[هم](?!\w))?"
)


def _plausible_date(m: re.Match) -> bool:
    if m.group("y2"):
        a, b = int(m.group("m2")), int(m.group("d2"))
    else:
        if m.group("s1") != "/" and len(m.group("y1")) != 4:
            return False
        a, b = int(m.group("m1")), int(m.group("d1"))
    # Accept day/month in either order; reject things like 45/99/2020.
    return (1 <= a <= 12 and 1 <= b <= 31) or (1 <= b <= 12 and 1 <= a <= 31)


def find_phones(text: str) -> list[Span]:
    norm, imap = normalize_with_map(text)
    spans = []
    for pat in _PHONE_PATTERNS:
        for m in pat.finditer(norm):
            spans.append(Span(*_to_original(text, imap, m.start(), m.end()), "PHONE", "regex"))
    return _dedupe(spans)


def find_national_ids(text: str) -> list[Span]:
    """Saudi national ID (starts with 1) or Iqama (starts with 2): ten digits.

    The checksum is not verified on purpose: for redaction, masking a number that
    merely looks like an ID is cheaper than letting a mistyped real one through.
    """
    norm, imap = normalize_with_map(text)
    spans = []
    for m in _ID_PATTERN.finditer(norm):
        # Digits written one by one ("1 2 3 4 ...") are usually a list. They are
        # taken as an ID only when a word such as "هوية" stands just before them.
        if len(_ID_SEPARATOR.split(m.group(0))) > _MAX_ID_GROUPS:
            window = norm[max(0, m.start() - _ID_CUE_WINDOW):m.start()].lower()
            if not any(cue in window for cue in _ID_CUES):
                continue
        spans.append(Span(*_to_original(text, imap, m.start(), m.end()), "NATIONAL_ID", "regex"))
    return spans


def find_mrns(text: str) -> list[Span]:
    """Medical record number: digits that follow an explicit cue ("MRN", "رقم الملف الطبي")."""
    norm, imap = normalize_with_map(text)
    spans = [
        Span(*_to_original(text, imap, m.start(1), m.end(1)), "MRN", "regex")
        for pattern in _MRN_PATTERNS
        for m in pattern.finditer(norm)
    ]
    return _dedupe(spans)


def find_emails(text: str) -> list[Span]:
    norm, imap = normalize_with_map(text)
    return [
        Span(*_to_original(text, imap, m.start(), m.end()), "EMAIL", "regex")
        for m in _EMAIL_PATTERN.finditer(norm)
    ]


def find_numeric_dates(text: str) -> list[Span]:
    norm, imap = normalize_with_map(text)
    return [
        Span(*_to_original(text, imap, m.start(), m.end()), "DATE", "regex")
        for m in _DATE_PATTERN.finditer(norm)
        if _plausible_date(m)
    ]


def _dedupe(spans: list[Span]) -> list[Span]:
    return sorted(set(spans))


def find_all(text: str) -> list[Span]:
    """Run every tool. IDs that overlap a phone, e-mail, MRN or date are dropped,
    because those more specific patterns explain the same digits."""
    if not text:
        return []
    specific = find_phones(text) + find_emails(text) + find_mrns(text) + find_numeric_dates(text)
    ids = [s for s in find_national_ids(text) if not any(s.overlaps(o) for o in specific)]
    return _dedupe(specific + ids)
