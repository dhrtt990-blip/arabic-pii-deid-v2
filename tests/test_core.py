"""Unit tests for text utilities, regex tools and the pipeline. All data is invented."""
import json

import pytest

from src import regex_tools
from src.common import (
    GoldFormatError,
    ModelOutputError,
    Span,
    find_occurrences,
    gold_spans,
    normalize_for_dedup,
    parse_model_output,
    redact,
    resolve_overlaps,
)
from src.pipeline import (
    STATUS_EMPTY,
    STATUS_MODEL_FAILURE,
    STATUS_OK,
    STATUS_REPAIRED,
    STATUS_TOO_LONG,
    STATUS_UNLOCATED,
    DeidPipeline,
    LLMUnavailable,
    MockClient,
    OllamaClient,
    ScriptedClient,
    check_leakage,
    run_model,
    split_chunks,
)


def reply(*entities):
    return json.dumps({"entities": [{"text": t, "type": y} for t, y in entities]}, ensure_ascii=False)


def texts(text, spans):
    return [text[s.start:s.end] for s in spans]


# ----------------------------- common ------------------------------------- #
class TestFindOccurrences:
    def test_name_is_not_matched_inside_longer_word(self):
        assert find_occurrences("سلّم عليه الطبيب", "علي") == []

    def test_standalone_name(self):
        text = "زار علي العيادة"
        assert texts(text, [Span(a, b, "PERS") for a, b in find_occurrences(text, "علي")]) == ["علي"]

    def test_attached_proclitics_are_tolerated(self):
        assert find_occurrences("تم الاتصال بمحمد", "محمد") == [(12, 16)]
        assert len(find_occurrences("وبمحمد", "محمد")) == 1

    def test_digits_inside_longer_number_do_not_match(self):
        assert find_occurrences("عام 2012", "12") == []

    def test_all_occurrences_returned(self):
        assert len(find_occurrences("الرياض ثم الرياض", "الرياض")) == 2

    def test_standalone_name_is_not_cut_out_of_a_longer_name(self):
        text = "حضر كريم مع ريم."
        assert [text[a:b] for a, b in find_occurrences(text, "ريم")] == ["ريم"]
        assert find_occurrences(text, "ريم")[0][0] == text.rindex("ريم")

    @pytest.mark.parametrize("text", [
        "المريض سلطان  الدوسري حضر",          # double space
        "المريض سلطـان الدوسري حضر",           # tatweel
        "المريض سُلطان الدوسري حضر",           # diacritic
        "المريض سلطان\nالدوسري حضر",          # line break inside the name
        "المريض سل\u200bطان الدوسري حضر",     # zero-width character
        "المريض سلطان الدوسرى حضر",            # final yaa written as alef maqsura
    ])
    def test_spelling_noise_does_not_hide_a_name(self, text):
        hits = find_occurrences(text, "سلطان الدوسري")
        assert len(hits) == 1
        a, b = hits[0]
        assert text[a:b].startswith("س") and text[b:].startswith(" حضر")

    def test_trailing_diacritic_stays_with_the_word(self):
        text = "قال محمدٌ ذلك"
        (a, b), = find_occurrences(text, "محمد")
        assert text[a:b] == "محمدٌ"


class TestParseModelOutput:
    def test_plain_json(self):
        out = parse_model_output(reply(("سلطان", "PERS")))
        assert out.entities == [{"text": "سلطان", "type": "PERS"}] and not out.cleaned

    def test_markdown_fence_is_removed_and_reported(self):
        out = parse_model_output("```json\n" + reply(("جدة", "LOC")) + "\n```")
        assert out.entities[0]["text"] == "جدة" and out.cleaned

    def test_think_block_is_removed(self):
        out = parse_model_output("<think>\n\n</think>\n\n" + reply())
        assert out.entities == [] and out.cleaned

    def test_alias_and_unknown_types(self):
        raw = json.dumps({"entities": [{"text": "أبها", "type": "GPE"}, {"text": "س", "type": "DRUG"}, {"text": "", "type": "PERS"}]})
        out = parse_model_output(raw)
        assert out.entities == [{"text": "أبها", "type": "LOC"}] and out.dropped_items == 2

    def test_duplicates_collapse(self):
        assert len(parse_model_output(reply(("ريم", "PERS"), ("ريم", "PERS"))).entities) == 1

    @pytest.mark.parametrize("raw", ["", "لا توجد كيانات", '{"entities": "none"}', '[{"text": "a"}]', '{"entities": [', None])
    def test_invalid_output_raises(self, raw):
        with pytest.raises(ModelOutputError):
            parse_model_output(raw)


class TestOverlapsAndRedaction:
    def test_nested_keeps_outer(self):
        out = resolve_overlaps([Span(0, 20, "ORG"), Span(5, 10, "PERS")])
        assert out == [Span(0, 20, "ORG")]

    def test_partial_overlap_is_merged_so_nothing_is_left_unmasked(self):
        out = resolve_overlaps([Span(0, 10, "PERS"), Span(5, 15, "LOC")])
        assert [(s.start, s.end) for s in out] == [(0, 15)]

    def test_regex_wins_on_identical_extent(self):
        out = resolve_overlaps([Span(0, 10, "DATE", "model"), Span(0, 10, "DATE", "regex")])
        assert out[0].source == "regex"

    def test_redact_uses_placeholders(self):
        text = "المريض سلطان في جدة"
        spans = [Span(7, 12, "PERS"), Span(16, 19, "LOC")]
        assert redact(text, spans) == "المريض [اسم] في [موقع]"


class TestGoldSpans:
    def test_repeated_mentions_take_successive_occurrences(self):
        row = {"id": "x", "text": "ريم ثم ريم", "entities": [{"text": "ريم", "type": "PERS"}, {"text": "ريم", "type": "PERS"}]}
        assert [(s.start, s.end) for s in gold_spans(row)] == [(0, 3), (7, 10)]

    def test_entity_missing_from_text(self):
        with pytest.raises(GoldFormatError):
            gold_spans({"id": "x", "text": "نص", "entities": [{"text": "غائب", "type": "PERS"}]})

    def test_mention_is_placed_on_the_whole_word_not_inside_another_word(self):
        row = {"id": "x", "text": "سلّم عليه الطبيب علي بحرارة.", "entities": [{"text": "علي", "type": "PERS"}]}
        (span,) = gold_spans(row)
        assert span.start == row["text"].rindex("علي") and span.start != row["text"].index("علي")

    def test_trailing_diacritic_is_part_of_the_gold_span_as_it_is_for_predictions(self):
        row = {"id": "x", "text": "حضر عليّ اليوم", "entities": [{"text": "علي", "type": "PERS"}]}
        (gold,) = gold_spans(row)
        (pred,) = find_occurrences(row["text"], "علي")
        assert (gold.start, gold.end) == pred

    def test_mention_inside_another_word_is_never_placed_silently(self):
        row = {"id": "x", "text": "علي قال عليه السلام", "entities": [{"text": "علي", "type": "PERS"}, {"text": "علي", "type": "PERS"}]}
        with pytest.raises(GoldFormatError, match="start"):
            gold_spans(row)

    def test_ambiguous_mentions_need_explicit_offsets(self):
        row = {"id": "x", "text": "ريم ثم ريم ثم ريم", "entities": [{"text": "ريم", "type": "PERS"}, {"text": "ريم", "type": "PERS"}]}
        with pytest.raises(GoldFormatError, match="start"):
            gold_spans(row)
        row["entities"] = [{"text": "ريم", "type": "PERS", "start": 0}, {"text": "ريم", "type": "PERS", "start": 14}]
        assert [(s.start, s.end) for s in gold_spans(row)] == [(0, 3), (14, 17)]

    def test_overlapping_gold_is_rejected(self):
        row = {"id": "x", "text": "مركز ناصر الطبي", "entities": [{"text": "مركز ناصر الطبي", "type": "ORG"}, {"text": "ناصر", "type": "PERS"}]}
        with pytest.raises(GoldFormatError):
            gold_spans(row)


def test_normalize_for_dedup_ignores_diacritics_digit_script_and_punctuation():
    assert normalize_for_dedup("مَوْعِدٌ: ١٢ مارس!") == normalize_for_dedup("موعد 12 مارس")


# ----------------------------- regex tools -------------------------------- #
class TestNationalId:
    @pytest.mark.parametrize("text,expected", [
        ("رقم الهوية 1087654321", "1087654321"),
        ("الإقامة 2098765432.", "2098765432"),
        ("الهوية ١٠٣٤٥٦٧٨٩٠", "١٠٣٤٥٦٧٨٩٠"),
        ("هويته 1 0 4 5 6 7 8 9 1 2 فقط", "1 0 4 5 6 7 8 9 1 2"),
        ("رقم 1012-345-678", "1012-345-678"),
        ("هوية:1087654321", "1087654321"),
        ("الهوية 1087654321/ والاسم", "1087654321"),
        ("الهوية 1087654321: سارية", "1087654321"),
        ("رقم 1 087-654-321", "1 087-654-321"),
    ])
    def test_found(self, text, expected):
        assert texts(text, regex_tools.find_national_ids(text)) == [expected]

    @pytest.mark.parametrize("text", [
        "الرقم 108765432",          # nine digits
        "الرقم 10876543210",        # eleven digits
        "الرقم 3087654321",         # does not start with 1 or 2
        "الموعد 2024-03-12 10:30",  # date and time
        "الغرفة 1015 في الدور 10",
        "الأرقام 1 2 3 4 5 6 7 8 9 10",     # a list of numbers, no cue word
        "قائمة 1-2-3-4-5-6-7-8-9-10",
        "تكرار 2 3 4 5 6 7 8 9 1 0 مرات",
    ])
    def test_not_found(self, text):
        assert regex_tools.find_national_ids(text) == []


class TestPhone:
    @pytest.mark.parametrize("text,expected", [
        ("جوال 0551234567", "0551234567"),
        ("على الرقم +966 50 123 4567 مساءً", "+966 50 123 4567"),
        ("00966551234567", "00966551234567"),
        ("966551234567", "966551234567"),
        ("جواله 05-5-123-4567.", "05-5-123-4567"),
        ("الجوال ٠٥٤١١١٢٢٣٣", "٠٥٤١١١٢٢٣٣"),
        ("جوال 055 - 123 - 4567", "055 - 123 - 4567"),
        ("جوال (055) 123 4567", "(055) 123 4567"),
        ("جوال +966 (0) 55 123 4567", "+966 (0) 55 123 4567"),
        ("وجوال:0551234567", "0551234567"),
    ])
    def test_found(self, text, expected):
        assert texts(text, regex_tools.find_phones(text)) == [expected]

    def test_zero_width_characters_do_not_hide_a_number(self):
        text = "الجوال ٠٥٥​١٢٣​٤٥٦٧ للتواصل"
        spans = regex_tools.find_phones(text)
        assert len(spans) == 1 and text[spans[0].start:spans[0].end].startswith("٠٥٥") and text[spans[0].end - 1] == "٧"

    @pytest.mark.parametrize("text", ["تحويلة 0551234", "جوال 055 123 456 (2)", "الغرفة 05 (12) 34 56 78 9"])
    def test_not_a_phone(self, text):
        assert regex_tools.find_phones(text) == []


class TestMrnEmailDate:
    @pytest.mark.parametrize("text,expected", [
        ("رقم الملف الطبي: MR-204518", "MR-204518"),
        ("MRN 445566", "445566"),
        ("mrn#7788990", "7788990"),
        ("ملف رقم 301245 للمراجع", "301245"),
        ("ورقم ملفه 4455667", "4455667"),
        ("رقم ملف 445566", "445566"),
        ("رقم ملف المريض 445566", "445566"),
        ("رقم الملف الطبي هو 445566", "445566"),
        ("الملف الطبي: 778899", "778899"),
    ])
    def test_mrn_needs_a_cue(self, text, expected):
        assert texts(text, regex_tools.find_mrns(text)) == [expected]

    @pytest.mark.parametrize("text", ["الغرفة 301245", "حُدّث الملف الطبي 2023", "السجل الطبي 2024 لا يحتوي على حساسية"])
    def test_number_without_a_specific_cue_is_not_an_mrn(self, text):
        assert regex_tools.find_mrns(text) == []

    def test_email(self):
        text = "راسلنا على n.alharbi@example.com اليوم"
        assert texts(text, regex_tools.find_emails(text)) == ["n.alharbi@example.com"]

    @pytest.mark.parametrize("text,expected", [
        ("بتاريخ 14/02/2025 راجع", "14/02/2025"),
        ("يوم 2025-04-09.", "2025-04-09"),
        ("الميلاد 12.03.1985", "12.03.1985"),
        ("في 15/09/1446هـ تم", "15/09/1446هـ"),
    ])
    def test_numeric_dates(self, text, expected):
        assert texts(text, regex_tools.find_numeric_dates(text)) == [expected]

    @pytest.mark.parametrize("text", ["النسبة 45/99/2020", "ضغط الدم 130/85", "الإصدار 1.2", "النسخة 1.2.20", "الجرعة 10.5.12", "12/03-2024", "جرعات 5-10-15 ملغ"])
    def test_non_dates(self, text):
        assert regex_tools.find_numeric_dates(text) == []


class TestFindAll:
    def test_phone_is_not_also_reported_as_id(self):
        types = [s.type for s in regex_tools.find_all("جوال 0551234567 وهوية 1087654321")]
        assert types == ["PHONE", "NATIONAL_ID"]

    def test_id_like_digits_inside_email_belong_to_the_email(self):
        text = "بريده saad.z1012345678@example.com"
        assert [s.type for s in regex_tools.find_all(text)] == ["EMAIL"]

    def test_offsets_refer_to_original_text(self):
        text = "الهويـة: ١٠٨٧٦٥٤٣٢١ والجوال 0551234567"
        for s in regex_tools.find_all(text):
            assert text[s.start:s.end] in ("١٠٨٧٦٥٤٣٢١", "0551234567")

    def test_empty(self):
        assert regex_tools.find_all("") == []


# ----------------------------- pipeline ----------------------------------- #
TEXT = "راجع المريض سلطان الدوسري مستشفى النخيل وهويته 1087654321 وجواله 0551234567."


class RaisingClient:
    name = "raising"

    def extract(self, text):
        raise LLMUnavailable("server down")


class TestPipeline:
    def test_happy_path_masks_model_and_regex_entities(self):
        client = ScriptedClient([reply(("سلطان الدوسري", "PERS"), ("مستشفى النخيل", "ORG"))])
        res = DeidPipeline(client).run(TEXT)
        assert res.status == STATUS_OK and res.released
        assert res.redacted == "راجع المريض [اسم] [جهة] وهويته [هوية] وجواله [جوال]."
        assert {e["source"] for e in res.entities} == {"model", "regex"}
        assert res.leak_findings == []

    def test_invalid_json_blocks_output_and_offers_regex_only_preview(self):
        res = DeidPipeline(ScriptedClient(["عذراً، لا أستطيع"])).run(TEXT)
        assert res.status == STATUS_MODEL_FAILURE and not res.released and res.redacted is None
        assert "[هوية]" in res.fallback_preview and "1087654321" not in res.fallback_preview
        assert res.model["ok"] is False

    def test_unreachable_model_blocks_output(self):
        res = DeidPipeline(RaisingClient()).run(TEXT)
        assert res.status == STATUS_MODEL_FAILURE and not res.released

    def test_regex_layer_works_when_model_finds_nothing(self):
        res = DeidPipeline(ScriptedClient([reply()])).run(TEXT)
        assert res.released and "1087654321" not in res.redacted and "0551234567" not in res.redacted

    def test_entity_the_model_named_but_is_nowhere_in_the_text_withholds_output(self):
        res = DeidPipeline(ScriptedClient([reply(("سلطان الدوسرى العتيبي", "PERS"))])).run(TEXT)
        assert res.status == STATUS_UNLOCATED and not res.released and res.redacted is None
        assert res.model["not_in_text"] == [{"text": "سلطان الدوسرى العتيبي", "type": "PERS"}]
        assert "[هوية]" in res.fallback_preview
        relaxed = DeidPipeline(ScriptedClient([reply(("سلطان الدوسرى العتيبي", "PERS"))]), block_on_unlocated=False).run(TEXT)
        assert relaxed.released

    @pytest.mark.parametrize("text", [
        "حضر سلطان  الدوسري اليوم.",
        "حضر سلطـان الدوسري اليوم.",
        "حضر سُلطان الدوسري اليوم.",
        "حضر سلطان\nالدوسري اليوم.",
        "حضر سل\u200bطان الدوسري اليوم.",
    ])
    def test_spelling_noise_in_the_text_does_not_let_a_name_through(self, text):
        res = DeidPipeline(ScriptedClient([reply(("سلطان الدوسري", "PERS"))])).run(text)
        assert res.status == STATUS_OK and res.released
        assert res.redacted == "حضر [اسم] اليوم."

    def test_name_with_written_case_ending_is_masked(self):
        res = DeidPipeline(ScriptedClient([reply(("محمد", "PERS"))])).run("قابلت محمدًا أمس.")
        assert res.status == STATUS_OK and res.released
        assert "محمد" not in res.redacted

    @pytest.mark.parametrize("text,entity", [
        ("راجع المريض مستشفى في الرياض ونُصح بممارسة الرياضة يومياً.", "الرياض"),
        ("حضر أمين من شركة التأمين.", "أمين"),
        ("حضر صالح وهو بحالة صالحة.", "صالح"),
        ("جاءت أمل وأمل الطبيب أن تتحسن.", "أمل"),
    ])
    def test_ordinary_words_that_contain_a_name_are_not_cut(self, text, entity):
        res = DeidPipeline(ScriptedClient([reply((entity, "PERS"))])).run(text)
        assert res.released and "[اسم]" in res.redacted
        for word in ("الرياضة", "التأمين", "صالحة"):
            assert (word in text) == (word in res.redacted)

    def test_name_glued_to_another_name_withholds_output(self):
        res = DeidPipeline(ScriptedClient([reply(("محمد", "PERS"), ("علي", "PERS"))])).run("زاره محمدعلي أمس.")
        assert res.status == STATUS_UNLOCATED and not res.released

    def test_short_name_inside_a_longer_name_is_left_alone(self):
        res = DeidPipeline(ScriptedClient([reply(("ريم", "PERS"))])).run("حضر كريم مع ريم.")
        assert res.status == STATUS_OK and res.redacted == "حضر كريم مع [اسم]."

    def test_second_mention_with_attached_particle_is_repaired(self):
        res = DeidPipeline(ScriptedClient([reply(("ريم", "PERS"))])).run("حضرت ريم ثم اتصلنا بريم لاحقاً.")
        assert res.status == STATUS_REPAIRED and res.redacted == "حضرت [اسم] ثم اتصلنا ب[اسم] لاحقاً."

    def test_identifier_glued_to_a_colon_is_masked(self):
        res = DeidPipeline(ScriptedClient([reply(("ماجد", "PERS"))])).run("المريض ماجد هوية:1087654321 وجوال:0551234567 ورقم ملفه 4455667")
        assert res.released
        assert res.redacted == "المريض [اسم] هوية:[هوية] وجوال:[جوال] ورقم ملفه [ملف طبي]"

    def test_leak_checker_repairs_mention_glued_to_other_letters(self):
        text = "حضر محمد إلى العيادة ثم اتصل أبومحمد لاحقاً."
        res = DeidPipeline(ScriptedClient([reply(("محمد", "PERS"))])).run(text)
        assert res.status == STATUS_REPAIRED and res.released
        assert "محمد" not in res.redacted
        assert any(e["source"] == "repair" for e in res.entities)

    def test_empty_and_too_long_inputs(self):
        pipe = DeidPipeline(ScriptedClient([reply()]), max_input_chars=50)
        assert pipe.run("   ").status == STATUS_EMPTY
        res = pipe.run("س" * 51)
        assert res.status == STATUS_TOO_LONG and not res.released and res.redacted is None

    def test_long_text_is_chunked_and_offsets_stay_correct(self):
        sentence = "تم تحويل المريض إلى الرياض للمتابعة. "
        text = sentence * 40
        res = DeidPipeline(MockClient("strong"), max_chunk_chars=200).run(text)
        assert res.released and "الرياض" not in res.redacted
        assert res.redacted.count("[موقع]") == 40

    def test_use_regex_false_is_model_only(self):
        res = DeidPipeline(ScriptedClient([reply()]), use_regex=False).run(TEXT)
        # The leak checker still runs its own regex pass and refuses to release.
        assert not res.released


class TestHelpers:
    def test_split_chunks_reassembles_and_respects_limit(self):
        text = "جملة أولى. " * 30 + "س" * 500
        chunks = split_chunks(text, 120)
        assert "".join(c for _, c in chunks) == text
        assert all(len(c) <= 120 for _, c in chunks)
        assert all(text[o:o + len(c)] == c for o, c in chunks)

    def test_placeholder_words_are_not_reported_as_leaks(self):
        assert check_leakage("اتصل على [جوال] وراجع [اسم]", ["جوال", "اسم"]) == []

    def test_check_leakage_flags_surviving_pattern_and_text(self):
        findings = check_leakage("اتصل على 0551234567 بخصوص [اسم] وسلطان الدوسري", ["سلطان الدوسري"])
        assert {f["kind"] for f in findings} == {"pattern", "entity_text"}

    def test_mock_weak_reply_is_fenced_and_parses(self):
        raw = MockClient("weak").extract("المريض محمد العتيبي")
        assert raw.startswith("```")
        assert parse_model_output(raw).entities == [{"text": "محمد العتيبي", "type": "PERS"}]

    def test_run_model_reports_cleaning(self):
        assert run_model(MockClient("weak"), "المريضة سارة").cleaned is True

    def test_ollama_payload_shape(self):
        base = OllamaClient("qwen3:1.7b", few_shot=True)._payload("نص")
        tuned = OllamaClient("pii-qwen3-lora")._payload("نص")
        assert base["think"] is False and base["stream"] is False and "format" in base
        assert base["options"]["temperature"] == 0
        assert len(base["messages"]) > len(tuned["messages"]) == 2
        assert tuned["messages"][-1] == {"role": "user", "content": "نص"}

    def test_ollama_unreachable_raises_llm_unavailable(self):
        with pytest.raises(LLMUnavailable):
            OllamaClient("x", host="http://127.0.0.1:9", timeout=2).extract("نص")
