"""The de-identification pipeline.

    text ──► LLM extractor (PERS / LOC / ORG / DATE as JSON)
         ──► regex tools  (national ID, phone, MRN, e-mail, numeric dates)
         ──► merger + redactor (overlap resolution, placeholders)
         ──► leakage checker   (blocks the output if anything sensitive survives)

Failure policy: the redacted text is released only when the model produced
valid JSON *and* the leakage checker passed. Otherwise the output is withheld
and the caller receives a status explaining why.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Protocol

from . import regex_tools
from .common import (
    PLACEHOLDERS,
    ModelOutputError,
    Span,
    find_occurrences,
    parse_model_output,
    redact,
    resolve_overlaps,
    spans_from_entities,
)
from .prompts import OUTPUT_SCHEMA, build_messages

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_BASE_MODEL = "qwen3:1.7b"
DEFAULT_TUNED_MODEL = "pii-qwen3-lora"

STATUS_OK = "OK"
STATUS_REPAIRED = "OK_AFTER_REPAIR"
STATUS_MODEL_FAILURE = "BLOCKED_MODEL_FAILURE"
STATUS_LEAK = "BLOCKED_LEAK"
STATUS_UNLOCATED = "BLOCKED_UNLOCATED_ENTITY"
STATUS_TOO_LONG = "BLOCKED_INPUT_TOO_LONG"
STATUS_EMPTY = "EMPTY_INPUT"

MESSAGES_AR = {
    STATUS_OK: "تم الإخفاء، ولم يرصد فاحص التسرّب أي معلومة حساسة متبقية.",
    STATUS_REPAIRED: "رصد فاحص التسرّب مقاطع متبقية وتم إخفاؤها تلقائياً ثم أُعيد الفحص بنجاح.",
    STATUS_MODEL_FAILURE: "تعذّر الحصول على مخرجات صالحة من النموذج. حُجب النص، والمعروض إخفاء بالقواعد فقط ويحتاج مراجعة بشرية.",
    STATUS_LEAK: "رصد فاحص التسرّب معلومة حساسة متبقية بعد الإخفاء. حُجب النص ويحتاج مراجعة بشرية.",
    STATUS_UNLOCATED: "ذكر النموذج مقطعاً حساساً تعذّر تحديد موضعه في النص، فقد يكون باقياً بصيغة أخرى. حُجب النص احتياطاً ويحتاج مراجعة بشرية.",
    STATUS_TOO_LONG: "النص أطول من الحد المسموح. قسّمه إلى أجزاء أصغر ثم أعد المحاولة.",
    STATUS_EMPTY: "لم يُدخل نص.",
}


# --------------------------------------------------------------------------- #
# LLM clients
# --------------------------------------------------------------------------- #
class LLMUnavailable(RuntimeError):
    """The model server could not be reached or returned an unusable reply."""


class LLMClient(Protocol):
    name: str

    def extract(self, text: str) -> str:
        """Return the raw model reply for one chunk of text."""


class OllamaClient:
    """Calls a model served locally by Ollama (`/api/chat`).

    `few_shot=True` is the baseline configuration: the untuned model with the
    full instruction plus worked examples. The tuned model uses `few_shot=False`,
    the format it was trained on.
    """

    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        few_shot: bool = False,
        use_schema: bool = True,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.few_shot = few_shot
        self.use_schema = use_schema
        self.timeout = timeout
        self.name = f"ollama:{model}" + ("+fewshot" if few_shot else "")

    def _payload(self, text: str) -> dict:
        payload = {
            "model": self.model,
            "messages": build_messages(text, few_shot=self.few_shot),
            "stream": False,
            "think": False,  # Qwen3 thinking mode costs hundreds of tokens and is not needed for extraction
            "options": {"temperature": 0, "seed": 0, "num_predict": 768},
        }
        if self.use_schema:
            payload["format"] = OUTPUT_SCHEMA  # constrained decoding: only schema-valid JSON can be generated
        return payload

    def extract(self, text: str) -> str:
        import requests

        payload = self._payload(text)
        url = f"{self.host}/api/chat"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 400 and "think" in resp.text.lower():
                # A model imported from GGUF may not be registered as a thinking model.
                payload.pop("think")
                resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise LLMUnavailable(f"cannot reach Ollama at {self.host}: {exc.__class__.__name__}") from None
        if resp.status_code != 200:
            raise LLMUnavailable(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            return resp.json()["message"]["content"]
        except (ValueError, KeyError, TypeError):
            raise LLMUnavailable("unexpected response shape from Ollama") from None

    def is_available(self) -> tuple[bool, str]:
        import requests

        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            names = {m.get("name", "") for m in resp.json().get("models", [])}
        except (requests.RequestException, ValueError):
            return False, f"Ollama is not reachable at {self.host}"
        wanted = {self.model, f"{self.model}:latest"}
        if not (names & wanted):
            return False, f"model '{self.model}' is not installed in Ollama"
        return True, "ok"


_AR = r"ء-ي"
_FIRST_NAMES = (
    "محمد أحمد عبدالله خالد سعد فهد سلطان ناصر تركي فيصل ماجد بدر سلمان "
    "نورة سارة فاطمة مريم ريم هند منى عائشة لمى نوف جواهر"
).split()
_CITIES = ["الرياض", "جدة", "مكة المكرمة", "المدينة المنورة", "الدمام", "الخبر", "أبها", "تبوك", "بريدة", "حائل", "الطائف", "نجران", "جازان"]
_MONTHS = (
    "يناير فبراير مارس أبريل مايو يونيو يوليو أغسطس سبتمبر أكتوبر نوفمبر ديسمبر "
    "محرم صفر رجب شعبان رمضان شوال"
).split()
_PERSON_CUE = re.compile(
    rf"(?:المريضة|المريض|الدكتورة|الدكتور|د\.|السيدة|السيد|الممرضة|الممرض|المراجعة|المراجع)\s+"
    rf"(?!(?:إلى|من|في|على|عن|مع|هو|هي|أن|إن|رقم|بتاريخ|يحتاج|يشكو|راجع|قادم|قادمة)(?![{_AR}]))"
    rf"([{_AR}]+(?:\s+ال[{_AR}]+)?)"
)
_NAME_PAIR = re.compile(rf"(?<![{_AR}])(?:{'|'.join(_FIRST_NAMES)})(?:\s+ال[{_AR}]+)?(?![{_AR}])")
_ORG_CUE = re.compile(rf"(?:مستشفى|مركز|مجمع|عيادات|جامعة|وزارة|شركة)(?:\s+ال[{_AR}]+){{1,3}}")
_TEXT_DATE = re.compile(rf"\d{{1,2}}\s+(?:{'|'.join(_MONTHS)})\s+\d{{4}}(?:\s?هـ)?|(?:{'|'.join(_MONTHS)})\s+\d{{4}}(?:\s?هـ)?")


class MockClient:
    """A rule-based stand-in for the model.

    It exists only so the pipeline, the evaluation script and the app can be
    exercised on a machine with no GPU and no Ollama. It is NOT a language
    model and its scores say nothing about the project's results.
    `level="weak"` finds fewer entities and wraps its reply in a markdown fence,
    which exercises the cleaning path and the before/after bookkeeping.
    """

    def __init__(self, level: str = "strong") -> None:
        if level not in ("weak", "strong"):
            raise ValueError("level must be 'weak' or 'strong'")
        self.level = level
        self.name = f"mock:{level}"

    def extract(self, text: str) -> str:
        found: list[tuple[int, str, str]] = []
        for m in _NAME_PAIR.finditer(text):
            found.append((m.start(), m.group(0), "PERS"))
        if self.level == "strong":
            for m in _PERSON_CUE.finditer(text):
                found.append((m.start(1), m.group(1), "PERS"))
            for m in _ORG_CUE.finditer(text):
                found.append((m.start(), m.group(0), "ORG"))
            for m in _TEXT_DATE.finditer(text):
                found.append((m.start(), m.group(0), "DATE"))
            for city in _CITIES:
                for a, _ in find_occurrences(text, city):
                    found.append((a, city, "LOC"))
        entities, seen = [], set()
        for _, etext, etype in sorted(found):
            if (etext, etype) not in seen:
                seen.add((etext, etype))
                entities.append({"text": etext, "type": etype})
        reply = json.dumps({"entities": entities}, ensure_ascii=False)
        return f"```json\n{reply}\n```" if self.level == "weak" else reply


class ScriptedClient:
    """Returns pre-written replies in order (used by the unit tests)."""

    def __init__(self, replies: list[str], name: str = "scripted") -> None:
        if not replies:
            raise ValueError("at least one reply is required")
        self.replies = list(replies)
        self.calls: list[str] = []
        self.name = name

    def extract(self, text: str) -> str:
        reply = self.replies[len(self.calls) % len(self.replies)]
        self.calls.append(text)
        return reply


def build_client(kind: str, model: str = DEFAULT_TUNED_MODEL, host: str = DEFAULT_HOST, few_shot: bool = False) -> LLMClient:
    """`kind` is "ollama", "mock" (strong) or "mock-weak"."""
    if kind == "ollama":
        return OllamaClient(model=model, host=host, few_shot=few_shot)
    if kind == "mock":
        return MockClient("strong")
    if kind == "mock-weak":
        return MockClient("weak")
    raise ValueError(f"unknown client kind: {kind!r}")


# --------------------------------------------------------------------------- #
# Model extraction (shared by the pipeline and the model-only evaluation)
# --------------------------------------------------------------------------- #
_SENTENCE_END = re.compile(r"(?<=[.!؟?\n])")


def split_chunks(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Split long text at sentence ends into (offset, chunk) pieces of at most
    `max_chars`. A single sentence longer than the limit is cut hard."""
    if len(text) <= max_chars:
        return [(0, text)]
    chunks: list[tuple[int, str]] = []
    start = pos = buffered = 0
    for piece in _SENTENCE_END.split(text):
        if not piece:
            continue
        if buffered and buffered + len(piece) > max_chars:
            chunks.append((start, text[start:pos]))
            start, buffered = pos, 0
        while len(piece) > max_chars:
            chunks.append((pos, text[pos:pos + max_chars]))
            pos += max_chars
            piece = piece[max_chars:]
            start = pos
        buffered += len(piece)
        pos += len(piece)
    if buffered:
        chunks.append((start, text[start:pos]))
    return chunks


@dataclass
class ModelExtraction:
    spans: list[Span] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)  # everything the model named, located or not
    ok: bool = True
    unavailable: bool = False                            # server or timeout failure, not a bad answer
    error: str | None = None
    cleaned: bool = False
    dropped_items: int = 0
    not_in_text: list[dict] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)


def run_model(client: LLMClient, text: str, max_chunk_chars: int = 1200) -> ModelExtraction:
    """Ask the model for entities, chunk by chunk, and locate them in `text`.

    There is deliberately no retry: decoding is greedy with a fixed seed, so a
    second identical request would return the same invalid reply.
    """
    result = ModelExtraction()
    for offset, chunk in split_chunks(text, max_chunk_chars):
        if not chunk.strip():
            continue
        try:
            raw = client.extract(chunk)
        except LLMUnavailable as exc:
            result.ok, result.unavailable, result.error = False, True, f"model unavailable: {exc}"
            return result
        result.raw.append(raw)
        try:
            parsed = parse_model_output(raw)
        except ModelOutputError as exc:
            result.ok, result.error = False, f"invalid model output: {exc}"
            return result
        result.cleaned = result.cleaned or parsed.cleaned
        result.dropped_items += parsed.dropped_items
        result.entities.extend(parsed.entities)
        spans, missing = spans_from_entities(chunk, parsed.entities, source="model")
        result.spans.extend(Span(s.start + offset, s.end + offset, s.type, s.source) for s in spans)
        result.not_in_text.extend(missing)
    result.spans = sorted(set(result.spans))
    return result


# --------------------------------------------------------------------------- #
# Leakage checker
# --------------------------------------------------------------------------- #
def check_leakage(redacted: str, entity_texts: list[str]) -> list[dict]:
    """Look for anything sensitive that survived redaction.

    1. Every regex tool is run again on the redacted text.
    2. Every entity string, whether it was masked or only named by the model,
       must not occur again in the output as a word of its own. A particle or a
       kunya word glued in front ("لمحمد", "أبومحمد") and a written case ending
       ("محمدًا") still count as the same word; diacritics, tatweel and extra
       spaces do not hide a match. A string that merely appears inside an
       unrelated longer word ("الرياض" in "الرياضة") is not a leak.
    """
    findings: list[dict] = []
    for sp in regex_tools.find_all(redacted):
        findings.append({"kind": "pattern", "type": sp.type, "text": redacted[sp.start:sp.end]})
    # Placeholders are our own text. Each is replaced by a single letter-like
    # character, so that a placeholder word is never reported as a leak and a
    # placeholder does not create a word boundary that the original text lacked.
    scrubbed = redacted
    for placeholder in PLACEHOLDERS.values():
        scrubbed = scrubbed.replace(placeholder, _SCRUB)
    for etext in sorted({t.strip() for t in entity_texts}):
        if len(etext) < 2:
            continue
        hits = find_occurrences(scrubbed, etext, "attached")
        if hits:
            findings.append({"kind": "entity_text", "text": etext, "count": len(hits)})
    return findings


_SCRUB = "\u2d30"  # a letter that never occurs in Arabic or Latin text


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@dataclass
class PipelineResult:
    status: str
    released: bool
    original: str
    redacted: str | None            # set only when the output is released
    fallback_preview: str | None    # regex-only redaction, for human review when the model failed
    entities: list[dict]            # disjoint spans with text / type / source
    model: dict                     # ok, error, cleaned, dropped_items, not_in_text, raw
    leak_findings: list[dict]
    message: str
    elapsed_s: float

    def spans(self) -> list[Span]:
        return [Span(e["start"], e["end"], e["type"], e["source"]) for e in self.entities]

    def to_dict(self) -> dict:
        return asdict(self)


class DeidPipeline:
    def __init__(
        self,
        client: LLMClient,
        use_regex: bool = True,
        max_chunk_chars: int = 1200,
        max_input_chars: int = 20000,
        block_on_unlocated: bool = True,
    ) -> None:
        self.client = client
        self.use_regex = use_regex
        self.max_chunk_chars = max_chunk_chars
        self.max_input_chars = max_input_chars
        self.block_on_unlocated = block_on_unlocated

    def _result(self, status, text, t0, *, released, redacted=None, fallback=None, spans=(), model=None, findings=()):
        return PipelineResult(
            status=status,
            released=released,
            original=text,
            redacted=redacted,
            fallback_preview=fallback,
            entities=[s.to_dict(text) for s in spans],
            model=model or {"ok": None, "error": None, "cleaned": False, "dropped_items": 0, "not_in_text": [], "raw": []},
            leak_findings=list(findings),
            message=MESSAGES_AR[status],
            elapsed_s=round(time.perf_counter() - t0, 3),
        )

    def run(self, text: str) -> PipelineResult:
        t0 = time.perf_counter()
        if not isinstance(text, str) or not text.strip():
            return self._result(STATUS_EMPTY, text if isinstance(text, str) else "", t0, released=True, redacted="")
        if len(text) > self.max_input_chars:
            return self._result(STATUS_TOO_LONG, text, t0, released=False)

        regex_spans = regex_tools.find_all(text) if self.use_regex else []
        extraction = run_model(self.client, text, self.max_chunk_chars)
        model_info = {
            "ok": extraction.ok,
            "unavailable": extraction.unavailable,
            "error": extraction.error,
            "cleaned": extraction.cleaned,
            "dropped_items": extraction.dropped_items,
            "not_in_text": extraction.not_in_text,
            "raw": extraction.raw,
        }

        if not extraction.ok:
            # Fallback guardrail: never release text the model did not screen.
            spans = resolve_overlaps(regex_spans)
            return self._result(
                STATUS_MODEL_FAILURE, text, t0, released=False,
                fallback=redact(text, spans), spans=spans, model=model_info,
            )

        spans = resolve_overlaps(extraction.spans + regex_spans)
        redacted = redact(text, spans)
        # The checker is given every string the model named, including the ones the
        # first pass could not locate: those are the ones most likely to be left behind.
        named = {e["text"]: e["type"] for e in extraction.entities}

        def check(current_spans: list[Span], current_redacted: str) -> list[dict]:
            return check_leakage(current_redacted, [text[s.start:s.end] for s in current_spans] + list(named))

        findings = check(spans, redacted)
        status = STATUS_OK

        if findings:
            # Repair in the coordinates of the original text: mask every remaining
            # occurrence of an entity string that the checker found.
            extra: list[Span] = []
            for f in findings:
                if f["kind"] != "entity_text":
                    continue
                etype = named.get(f["text"]) or next((s.type for s in spans if text[s.start:s.end].strip() == f["text"]), "PERS")
                for a, b in find_occurrences(text, f["text"], "attached"):
                    extra.append(Span(a, b, etype, "repair"))
            if extra:
                spans = resolve_overlaps(spans + extra)
                redacted = redact(text, spans)
                findings = check(spans, redacted)
                status = STATUS_REPAIRED
            if findings:
                return self._result(STATUS_LEAK, text, t0, released=False, spans=spans, model=model_info, findings=findings)

        # The model called a string sensitive and it could not be placed on any word
        # of the text: it altered the name while copying it, or the name is glued to
        # another word. Either way something sensitive may still be visible, so the
        # output is withheld for a person to look at.
        if self.block_on_unlocated and extraction.not_in_text:
            return self._result(
                STATUS_UNLOCATED, text, t0, released=False,
                fallback=redacted, spans=spans, model=model_info,
            )

        return self._result(status, text, t0, released=True, redacted=redacted, spans=spans, model=model_info)
