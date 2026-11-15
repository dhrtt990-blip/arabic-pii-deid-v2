"""Unit tests for data preparation, the overlap guarantee and golden-test validation."""
import json
import random
from pathlib import Path

import pytest

from src.common import MODEL_TYPES, normalize_for_dedup, parse_model_output, read_jsonl, write_jsonl
from src.data_prep import (
    ANERCORP_TYPE_MAP,
    WOJOOD_TYPE_MAP,
    GoldenIndex,
    build_parser,
    cap_negatives,
    clean_examples,
    detokenize,
    generate_synthetic,
    load_anercorp_file,
    load_wojood,
    prepare,
    tagged_sentence_to_example,
)
from src.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from src.validate_golden import main as validate_main
from src.validate_golden import validate_rows

ROOT = Path(__file__).resolve().parents[1]

GOLDEN = [
    {"id": "g1", "text": "راجع المريض سلطان الدوسري مستشفى النخيل بتاريخ 14/02/2025.",
     "entities": [{"text": "سلطان الدوسري", "type": "PERS"}, {"text": "مستشفى النخيل", "type": "ORG"}, {"text": "14/02/2025", "type": "DATE"}],
     "written_by": "A", "checked_by": "B"},
    {"id": "g2", "text": "رقم الهوية 1087654321 والجوال 0551234567.",
     "entities": [{"text": "1087654321", "type": "NATIONAL_ID"}, {"text": "0551234567", "type": "PHONE"}],
     "written_by": "B", "checked_by": "A"},
    {"id": "g3", "text": "يُنصح المريض بشرب السوائل والراحة التامة.", "entities": [], "written_by": "A", "checked_by": "C"},
]

WOJOOD_SAMPLE = """جامعة B-ORG
بيرزيت I-ORG B-GPE
في O
فلسطين B-GPE
. O

$#$ O
زار O
سامر B-PERS
الخطيب I-PERS
القدس B-GPE
عام O
1987 B-DATE
"""

ANERCORP_SAMPLE = """زار O
سامر B-PERS
الخطيب I-PERS
مدينة O
نابلس B-LOC
. O
وأعلنت O
شركة B-ORG
الأفق I-ORG
عن O
منتج B-MISC
جديد O
. O
"""


# ----------------------------- conversion --------------------------------- #
def test_detokenize_attaches_punctuation_and_keeps_offsets():
    text, offsets = detokenize(["قال", "(", "سامر", ")", "،", "نعم", "."])
    assert text == "قال (سامر)، نعم."
    assert [text[a:b] for a, b in offsets] == ["قال", "(", "سامر", ")", "،", "نعم", "."]


def test_nested_tags_reduce_to_outermost_entity():
    ex = tagged_sentence_to_example(
        ["جامعة", "بيرزيت", "في", "فلسطين"],
        [["B-ORG"], ["I-ORG", "B-GPE"], ["O"], ["B-GPE"]],
        WOJOOD_TYPE_MAP, "wojood",
    )
    assert ex["entities"] == [{"text": "جامعة بيرزيت", "type": "ORG"}, {"text": "فلسطين", "type": "LOC"}]


def test_unmapped_types_are_dropped_and_adjacent_entities_split():
    ex = tagged_sentence_to_example(
        ["الوزير", "سامر", "الخطيب", "نابلس", "جنين"],
        [["B-OCC"], ["B-PERS"], ["I-PERS"], ["B-GPE"], ["B-GPE"]],
        WOJOOD_TYPE_MAP, "wojood",
    )
    assert ex["entities"] == [
        {"text": "سامر الخطيب", "type": "PERS"}, {"text": "نابلس", "type": "LOC"}, {"text": "جنين", "type": "LOC"},
    ]


def test_wojood_file_reader_handles_blank_lines_and_document_separator(tmp_path):
    path = tmp_path / "w.txt"
    path.write_text(WOJOOD_SAMPLE, encoding="utf-8")
    examples = load_wojood([str(path)])
    assert [e["text"] for e in examples] == ["جامعة بيرزيت في فلسطين.", "زار سامر الخطيب القدس عام 1987"]
    # A year on its own is tagged DATE in the source but is not a target under our guidelines.
    assert examples[1]["entities"] == [{"text": "سامر الخطيب", "type": "PERS"}, {"text": "القدس", "type": "LOC"}]
    assert examples[1]["vague_dates_dropped"] == 1


@pytest.mark.parametrize("text,expected", [
    ("15 أيار 1987", True), ("آذار 2021", True), ("14/02/2025", True), ("3 رمضان 1446هـ", True),
    ("1987", False), ("اليوم", False), ("الأسبوع الماضي", False), ("مارس", False),
])
def test_only_specific_dates_are_kept_as_targets(text, expected):
    from src.data_prep import is_specific_date
    assert is_specific_date(text) is expected


def test_specific_date_in_tagged_data_is_kept():
    ex = tagged_sentence_to_example(
        ["وُلد", "في", "15", "أيار", "1987", "."],
        [["O"], ["O"], ["B-DATE"], ["I-DATE"], ["I-DATE"], ["O"]],
        WOJOOD_TYPE_MAP, "wojood",
    )
    assert ex["entities"] == [{"text": "15 أيار 1987", "type": "DATE"}] and ex["vague_dates_dropped"] == 0


def test_anercorp_rows_are_grouped_into_sentences_and_misc_dropped(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text(ANERCORP_SAMPLE, encoding="utf-8")
    examples = load_anercorp_file(str(path))
    assert len(examples) == 2
    assert examples[0]["entities"] == [{"text": "سامر الخطيب", "type": "PERS"}, {"text": "نابلس", "type": "LOC"}]
    assert examples[1]["entities"] == [{"text": "شركة الأفق", "type": "ORG"}]
    assert set(ANERCORP_TYPE_MAP.values()) <= set(MODEL_TYPES)


# ----------------------------- synthetic ---------------------------------- #
def test_synthetic_is_reproducible_and_consistent():
    a, b, c = generate_synthetic(120, seed=1), generate_synthetic(120, seed=1), generate_synthetic(120, seed=2)
    assert a == b and a != c
    assert any(not ex["entities"] for ex in a) and any(ex["entities"] for ex in a)
    for ex in a:
        assert ex["source"] == "template"
        for ent in ex["entities"]:
            assert ent["type"] in MODEL_TYPES and ent["text"] in ex["text"]


def test_clean_examples_and_negative_cap():
    examples = [
        {"text": "جملة قصيرة", "entities": [], "source": "x"},
        {"text": "هذه جملة مكررة تماماً", "entities": [], "source": "x"},
        {"text": "هذه  جملة مكررة تماماً!", "entities": [], "source": "x"},
    ]
    assert len(clean_examples(examples)) == 1
    pool = [{"text": f"p{i}", "entities": [{"text": "p", "type": "PERS"}], "source": "x"} for i in range(30)]
    pool += [{"text": f"n{i}", "entities": [], "source": "x"} for i in range(30)]
    capped = cap_negatives(pool, 0.25, random.Random(0))
    assert sum(1 for e in capped if not e["entities"]) == 10


# ----------------------------- overlap ------------------------------------ #
class TestGoldenIndex:
    def test_exact_match_survives_cosmetic_edits(self):
        hit = GoldenIndex(GOLDEN, 0.8).match("راجَع المريض  سلطان الدوسري مستشفى النخيل بتاريخ ١٤/٠٢/٢٠٢٥ !")
        assert hit == {"golden_id": "g1", "kind": "exact", "score": 1.0}

    def test_golden_sentence_inside_a_longer_one(self):
        hit = GoldenIndex(GOLDEN, 0.8).match("ملاحظة: راجع المريض سلطان الدوسري مستشفى النخيل بتاريخ 14/02/2025. ثم غادر مساءً.")
        assert hit and hit["kind"] == "contained"

    def test_near_duplicate(self):
        hit = GoldenIndex(GOLDEN, 0.7).match("راجع المريض سلطان الدوسري مستشفى النخيل بتاريخ 14/02/2026.")
        assert hit and hit["kind"] == "near_duplicate" and hit["golden_id"] == "g1"

    def test_edited_copy_with_an_added_clause_is_caught(self):
        edited = "زار المريض سلطان الدوسري مستشفى النخيل بتاريخ 14/02/2025 وكان يشكو من صداع شديد منذ ثلاثة أيام متواصلة."
        hit = GoldenIndex(GOLDEN, 0.8).match(edited)
        assert hit and hit["golden_id"] == "g1" and hit["kind"] in ("shared_content", "near_duplicate")

    def test_short_golden_sentence_inside_a_longer_training_sentence(self):
        rows = GOLDEN + [{"id": "g4", "text": "اتصل بسعد غداً.", "entities": []}]
        hit = GoldenIndex(rows, 0.8).match("قال الطبيب للممرضة: اتصل بسعد غداً بعد الظهر لتأكيد الموعد.")
        assert hit and hit["golden_id"] == "g4"

    def test_unrelated_sentence_passes(self):
        assert GoldenIndex(GOLDEN, 0.8).match("تم صرف مضاد حيوي ومسكن للألم مع تعليمات العناية بالجرح.") is None


def _args(tmp_path, golden_rows, extra=()):
    golden = tmp_path / "golden.jsonl"
    write_jsonl(golden, golden_rows)
    wojood = tmp_path / "w.txt"
    wojood.write_text(WOJOOD_SAMPLE, encoding="utf-8")
    aner = tmp_path / "a.txt"
    aner.write_text(ANERCORP_SAMPLE, encoding="utf-8")
    team = tmp_path / "team.jsonl"
    write_jsonl(team, [
        {"id": "leak", "text": "راجَع المريض سلطان الدوسري مستشفى النخيل بتاريخ 14/02/2025", "entities": [{"text": "سلطان الدوسري", "type": "PERS"}]},
        {"id": "ok", "text": "المريضة وعد الجهني هويتها 1011111111 تسكن في ينبع.",
         "entities": [{"text": "وعد الجهني", "type": "PERS"}, {"text": "1011111111", "type": "NATIONAL_ID"}, {"text": "ينبع", "type": "LOC"}]},
    ])
    argv = ["--golden", str(golden), "--wojood", str(wojood), "--anercorp-file", str(aner), "--team-jsonl", str(team),
            "--synthetic-n", "40", "--out", str(tmp_path / "out"), "--min-golden-rows", "0", *extra]
    return build_parser().parse_args(argv)


class TestPrepare:
    def test_end_to_end_removes_golden_overlap_and_writes_chat_rows(self, tmp_path):
        report = prepare(_args(tmp_path, GOLDEN))
        assert report["removed_for_golden_overlap"] >= 1 and report["overlap_after_filtering"] == 0
        assert any(r["golden_id"] == "g1" for r in report["removed_examples"])
        rows = read_jsonl(tmp_path / "out" / "train.jsonl") + read_jsonl(tmp_path / "out" / "val.jsonl")
        assert len(rows) == report["train_rows"] + report["val_rows"] and report["val_rows"] >= 1
        golden_norms = {normalize_for_dedup(g["text"]) for g in GOLDEN}
        for row in rows:
            roles = [m["role"] for m in row["messages"]]
            assert roles == ["system", "user", "assistant"]
            assert row["messages"][0]["content"] == SYSTEM_PROMPT
            assert normalize_for_dedup(row["messages"][1]["content"]) not in golden_norms
            parsed = parse_model_output(row["messages"][2]["content"])
            assert not parsed.cleaned and parsed.dropped_items == 0

    def test_team_identifiers_are_not_model_targets(self, tmp_path):
        prepare(_args(tmp_path, GOLDEN))
        rows = read_jsonl(tmp_path / "out" / "train.jsonl") + read_jsonl(tmp_path / "out" / "val.jsonl")
        team = [r for r in rows if r["source"] == "team"]
        assert len(team) == 1
        assert json.loads(team[0]["messages"][2]["content"])["entities"] == [
            {"text": "وعد الجهني", "type": "PERS"}, {"text": "ينبع", "type": "LOC"},
        ]

    def test_report_records_golden_hash(self, tmp_path):
        report = prepare(_args(tmp_path, GOLDEN))
        saved = json.loads((tmp_path / "out" / "prep_report.json").read_text(encoding="utf-8"))
        assert saved["golden_sha256"] == report["golden_sha256"] and len(saved["golden_sha256"]) == 64

    def test_missing_golden_file_stops_everything(self, tmp_path):
        args = _args(tmp_path, GOLDEN)
        args.golden = str(tmp_path / "absent.jsonl")
        with pytest.raises(SystemExit):
            prepare(args)

    def test_unreviewed_golden_is_refused_by_default(self, tmp_path):
        rows = [{**g, "checked_by": "TBD"} for g in GOLDEN]
        with pytest.raises(SystemExit):
            prepare(_args(tmp_path, rows))
        prepare(_args(tmp_path, rows, extra=["--allow-unreviewed"]))

    def test_few_shot_example_in_golden_is_detected(self, tmp_path):
        rows = GOLDEN + [{"id": "g4", "text": FEW_SHOT_EXAMPLES[1][0], "entities": [], "written_by": "A", "checked_by": "B"}]
        with pytest.raises(SystemExit, match="few-shot"):
            prepare(_args(tmp_path, rows))


# ----------------------------- golden validation -------------------------- #
class TestValidateGolden:
    def test_valid_rows_pass(self):
        errors, _, stats = validate_rows(GOLDEN, min_rows=3)
        assert errors == [] and stats["rows"] == 3 and stats["rows_without_entities"] == 1
        assert stats["entities_by_type"]["PERS"] == 1

    def test_min_rows(self):
        errors, _, _ = validate_rows(GOLDEN)
        assert any("at least 100" in e for e in errors)

    @pytest.mark.parametrize("patch,needle", [
        ({"id": "g1"}, "duplicate id"),
        ({"entities": [{"text": "سلطان", "type": "NAME"}]}, "unknown type"),
        ({"entities": [{"text": "غير موجود", "type": "PERS"}]}, "not found"),
        ({"entities": [{"text": " سلطان", "type": "PERS"}]}, "leading or trailing"),
        ({"entities": "none"}, "must be a list"),
        ({"checked_by": "TBD"}, "must name"),
        ({"checked_by": "A", "written_by": "A"}, "other than its writer"),
        ({"text": GOLDEN[0]["text"]}, "same text"),
    ])
    def test_errors(self, patch, needle):
        bad = {"id": "g9", "text": "حضر سلطان اليوم.", "entities": [], "written_by": "A", "checked_by": "B", **patch}
        errors, _, _ = validate_rows(GOLDEN + [bad], min_rows=0)
        assert any(needle in e for e in errors), errors

    def test_overlapping_gold_entities_are_an_error(self):
        bad = {"id": "g9", "text": "مركز ناصر الطبي", "written_by": "A", "checked_by": "B",
               "entities": [{"text": "مركز ناصر الطبي", "type": "ORG"}, {"text": "ناصر", "type": "PERS"}]}
        errors, _, _ = validate_rows([bad], min_rows=0)
        assert any("overlap" in e for e in errors)

    def test_unannotated_identifier_is_a_warning(self):
        row = {"id": "g9", "text": "الهوية 1087654321", "entities": [], "written_by": "A", "checked_by": "B"}
        _, warnings, _ = validate_rows([row], min_rows=0)
        assert any("NATIONAL_ID" in w for w in warnings)

    def test_template_is_a_valid_draft_but_not_a_finished_golden_test(self):
        rows = read_jsonl(ROOT / "data" / "golden_test_template.jsonl")
        errors, warnings, _ = validate_rows(rows, min_rows=0, require_review=False)
        assert errors == [] and not any("looks like" in w for w in warnings)
        strict_errors, _, _ = validate_rows(rows)
        assert any("at least 100" in e for e in strict_errors) and any("must name" in e for e in strict_errors)

    def test_cli_exit_codes(self, tmp_path, capsys):
        good = tmp_path / "good.jsonl"
        write_jsonl(good, GOLDEN)
        assert validate_main([str(good), "--min-rows", "3"]) == 0
        assert validate_main([str(good)]) == 1
        assert validate_main([str(tmp_path / "absent.jsonl")]) == 1
        assert "FAILED" in capsys.readouterr().out
