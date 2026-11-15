"""Unit tests for the evaluation script, the safety check and the Streamlit app."""
import json
from pathlib import Path

import pytest

from src.common import Span, write_jsonl
from src.evaluate import classify_example, count_transitions, predict_full, predict_model_only, score_system
from src.evaluate import main as eval_main
from src.pipeline import STATUS_MODEL_FAILURE, STATUS_TOO_LONG, DeidPipeline, LLMUnavailable, MockClient, ScriptedClient
from src.safety_check import load_inputs, run_checks
from src.safety_check import main as safety_main

ROOT = Path(__file__).resolve().parents[1]

GOLDEN = [
    {"id": "g1", "text": "راجع سلطان الدوسري مستشفى النخيل وهويته 1087654321.",
     "entities": [{"text": "سلطان الدوسري", "type": "PERS"}, {"text": "مستشفى النخيل", "type": "ORG"}, {"text": "1087654321", "type": "NATIONAL_ID"}],
     "written_by": "A", "checked_by": "B"},
    {"id": "g2", "text": "يُنصح بشرب السوائل والراحة.", "entities": [], "written_by": "B", "checked_by": "A"},
    {"id": "g3", "text": "تم تحويل المريضة ريم الشهري إلى الرياض.",
     "entities": [{"text": "ريم الشهري", "type": "PERS"}, {"text": "الرياض", "type": "LOC"}],
     "written_by": "A", "checked_by": "C"},
]


def reply(*entities):
    return json.dumps({"entities": [{"text": t, "type": y} for t, y in entities]}, ensure_ascii=False)


# Truncated name (boundary error) + correct ORG; a spurious entity; a wrong type + a miss.
SCRIPT = [
    reply(("سلطان", "PERS"), ("مستشفى النخيل", "ORG")),
    reply(("السوائل", "PERS")),
    reply(("ريم الشهري", "ORG")),
]


def by_id(rows):
    return {r["id"]: r for r in rows}


# ----------------------------- classification ----------------------------- #
class TestClassify:
    def test_exact_match(self):
        r = classify_example([Span(0, 5, "PERS", "gold")], [Span(0, 5, "PERS")])
        assert r["tp"] == 1 and not r["pred_errors"] and not r["gold_errors"]

    def test_boundary_error(self):
        r = classify_example([Span(0, 10, "PERS", "gold")], [Span(0, 5, "PERS")])
        assert r["tp"] == 0 and r["pred_errors"][0][1] == "boundary_error" and r["gold_errors"][0][1] == "boundary_error"

    def test_wrong_type(self):
        r = classify_example([Span(0, 5, "PERS", "gold")], [Span(0, 5, "ORG")])
        assert r["pred_errors"][0][1] == "wrong_type" and r["gold_errors"][0][1] == "wrong_type"

    def test_spurious_and_missed(self):
        r = classify_example([Span(0, 5, "PERS", "gold")], [Span(20, 25, "LOC")])
        assert r["pred_errors"][0][1] == "false_positive" and r["gold_errors"][0][1] == "false_negative"


# ----------------------------- scoring ------------------------------------ #
class TestScoring:
    def test_model_only_scores_and_error_types(self):
        client = ScriptedClient(SCRIPT)
        preds = by_id([predict_model_only(client, ex) for ex in GOLDEN])
        s = score_system(GOLDEN, preds, "model_types")
        assert (s["tp"], s["fp"], s["fn"]) == (1, 3, 3)
        assert s["precision"] == 0.25 and s["recall"] == 0.25
        assert s["errors"] == {"invalid_json_examples": 0, "boundary_error": 1, "wrong_type": 1, "false_positive": 1, "false_negative": 1}
        # Leaked: the half-masked name in g1 and the city in g3. The mistyped name in g3 is still masked.
        assert s["leaked_entities"] == 2 and s["gold_entities"] == 4
        assert s["examples_fully_correct"] == 0
        assert s["by_type"]["ORG"]["tp"] == 1 and s["by_type"]["ORG"]["fp"] == 1

    def test_identifier_leaks_without_the_regex_layer_and_not_with_it(self):
        client, pipeline = ScriptedClient(SCRIPT), DeidPipeline(ScriptedClient(SCRIPT))
        model_only = by_id([predict_model_only(client, ex) for ex in GOLDEN])
        full = by_id([predict_full(pipeline, ex) for ex in GOLDEN])
        a = score_system(GOLDEN, model_only, "all_types")
        b = score_system(GOLDEN, full, "all_types")
        assert a["by_type"]["NATIONAL_ID"]["recall"] == 0.0 and b["by_type"]["NATIONAL_ID"]["recall"] == 1.0
        # Model only leaks the half-masked name, the ID and the city; the system masks the ID.
        assert (a["leaked_entities"], b["leaked_entities"]) == (3, 2)
        assert b["examples_withheld"] == 0

    def test_invalid_json_leaks_everything_for_model_only_but_is_withheld_by_the_system(self):
        bad = ScriptedClient(["عذراً"])
        model_only = by_id([predict_model_only(bad, ex) for ex in GOLDEN])
        full_rows = [predict_full(DeidPipeline(ScriptedClient(["عذراً"])), ex) for ex in GOLDEN]
        assert all(r["status"] == STATUS_MODEL_FAILURE for r in full_rows)
        a = score_system(GOLDEN, model_only, "all_types")
        b = score_system(GOLDEN, by_id(full_rows), "all_types")
        assert a["errors"]["invalid_json_examples"] == 3 and a["leakage_rate"] == 1.0
        assert b["leaked_entities"] == 0 and b["examples_withheld"] == 3 and b["examples_fully_correct"] == 0
        # Withheld text is not a success on the leak measure: nothing was released, so nothing is counted.
        assert b["gold_entities_in_released_text"] == 0
        assert not any(v["released_leak_free"] for v in b["per_example"].values())

    def test_server_failure_is_not_scored_as_invalid_json(self, tmp_path):
        class Down:
            name = "down"

            def extract(self, text):
                raise LLMUnavailable("timeout")

        row = predict_model_only(Down(), GOLDEN[0])
        assert row["model_unavailable"] is True and row["invalid_json"] is False
        assert predict_full(DeidPipeline(Down()), GOLDEN[0])["model_unavailable"] is True

    def test_transitions(self):
        before = {"a": {"correct": False}, "b": {"correct": True}, "c": {"correct": True}, "d": {"correct": False}}
        after = {"a": {"correct": True}, "b": {"correct": False}, "c": {"correct": True}, "d": {"correct": False}}
        t = count_transitions(before, after, "correct")
        assert (t["wrong_to_right"], t["right_to_wrong"], t["right_in_both"], t["wrong_in_both"]) == (1, 1, 1, 1)
        assert t["wrong_to_right_ids"] == ["a"] and t["right_to_wrong_ids"] == ["b"]


# ----------------------------- evaluate CLI ------------------------------- #
class TestEvaluateCli:
    def test_mock_dry_run_writes_marked_report_and_rescoring_matches(self, tmp_path):
        golden = tmp_path / "golden.jsonl"
        write_jsonl(golden, GOLDEN)
        out = tmp_path / "out"
        assert eval_main(["--golden", str(golden), "--client", "mock", "--min-rows", "0", "--out", str(out)]) == 0
        report = (out / "results.md").read_text(encoding="utf-8")
        assert "DRY RUN WITH THE MOCK CLIENT" in report and "Examples that changed" in report and "SHA-256" in report
        results = json.loads((out / "results.json").read_text(encoding="utf-8"))
        assert set(results["scores"]) == {f"{s}/{scope}" for s in ("baseline", "tuned", "full") for scope in ("model_types", "all_types")}
        assert len(results["transitions"]) == 3 and results["meta"]["golden_rows"] == 3
        for system in ("baseline", "tuned", "full"):
            assert (out / f"predictions_{system}.jsonl").exists() and (out / f"errors_{system}.jsonl").exists()

        out2 = tmp_path / "rescored"
        assert eval_main(["--golden", str(golden), "--client", "mock", "--min-rows", "0", "--out", str(out2), "--from-predictions", str(out)]) == 0
        rescored = json.loads((out2 / "results.json").read_text(encoding="utf-8"))
        assert rescored["scores"] == results["scores"]

    def test_invalid_golden_is_refused(self, tmp_path):
        golden = tmp_path / "golden.jsonl"
        write_jsonl(golden, [{**g, "checked_by": "TBD"} for g in GOLDEN])
        assert eval_main(["--golden", str(golden), "--client", "mock", "--min-rows", "0", "--out", str(tmp_path / "o")]) == 1
        assert eval_main(["--golden", str(tmp_path / "absent.jsonl"), "--client", "mock"]) == 1

    def test_unreachable_ollama_stops_with_a_clear_message(self, tmp_path):
        golden = tmp_path / "golden.jsonl"
        write_jsonl(golden, GOLDEN)
        with pytest.raises(SystemExit, match="Ollama"):
            eval_main(["--golden", str(golden), "--min-rows", "0", "--host", "http://127.0.0.1:9", "--out", str(tmp_path / "o")])


# ----------------------------- safety check ------------------------------- #
class TestSafetyCheck:
    def test_shipped_inputs_are_ten_and_varied(self):
        rows = load_inputs(ROOT / "data" / "safety_inputs.jsonl")
        assert len(rows) == 10 and len({r["category"] for r in rows}) >= 8
        assert len(by_id(rows)["s10"]["text"]) > 20000

    def test_outcomes_are_reported_including_failures(self):
        rows = load_inputs(ROOT / "data" / "safety_inputs.jsonl")
        outcomes = by_id(run_checks(DeidPipeline(MockClient("strong")), rows))
        assert outcomes["s04"]["verdict"] == "PASS" and outcomes["s05"]["verdict"] == "PASS"   # regex layer
        assert outcomes["s10"]["status"] == STATUS_TOO_LONG and outcomes["s10"]["verdict"] == "WITHHELD"
        assert outcomes["s06"]["verdict"] == "FAIL" and outcomes["s06"]["leaked"]             # digits spelled as words

    def test_too_few_inputs_or_missing_targets_are_rejected(self, tmp_path):
        path = tmp_path / "few.jsonl"
        write_jsonl(path, [{"id": "x", "category": "a", "text": "نص", "must_not_appear": ["نص"]}])
        with pytest.raises(ValueError, match="at least 10"):
            load_inputs(path)
        write_jsonl(path, [{"id": "x", "category": "a", "text": "نص"}])
        with pytest.raises(ValueError, match="must_not_appear"):
            load_inputs(path)

    def test_cli_mock_run_is_marked_as_dry_run(self, tmp_path):
        assert safety_main(["--inputs", str(ROOT / "data" / "safety_inputs.jsonl"), "--client", "mock", "--out", str(tmp_path)]) == 0
        report = (tmp_path / "safety_report.md").read_text(encoding="utf-8")
        assert "DRY RUN WITH THE MOCK CLIENT" in report and report.count("\n| s") == 10


# ----------------------------- Streamlit app ------------------------------ #
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(ROOT / "app" / "streamlit_app.py")


def markdown_of(at) -> str:
    return "\n".join(m.value for m in at.markdown)


class TestApp:
    def test_mock_engine_redacts_and_shows_sources(self, monkeypatch):
        monkeypatch.setenv("PII_APP_ENGINE", "mock")
        at = AppTest.from_file(APP, default_timeout=30).run()
        assert not at.exception
        assert "direction: rtl" in markdown_of(at)
        at.button(key="run_btn").click().run()
        assert not at.exception and len(at.success) == 1
        md = markdown_of(at)
        assert "[هوية]" in md and "[جوال]" in md and "[اسم]" in md
        assert "القواعد (Regex)" in md and "النموذج" in md

    def test_user_text_is_escaped(self, monkeypatch):
        monkeypatch.setenv("PII_APP_ENGINE", "mock")
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.text_area(key="input_text").set_value("<script>alert(1)</script> المريض محمد العتيبي").run()
        at.button(key="run_btn").click().run()
        md = markdown_of(at)
        assert not at.exception and "<script>" not in md and "&lt;script&gt;" in md

    def test_blank_lines_in_user_text_cannot_break_out_of_the_text_box(self, monkeypatch):
        monkeypatch.setenv("PII_APP_ENGINE", "mock")
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.text_area(key="input_text").set_value("المريض محمد العتيبي\n\n# عنوان\n[رابط](http://example.com)").run()
        at.button(key="run_btn").click().run()
        boxes = [m.value for m in at.markdown if 'class="box"' in m.value]
        assert not at.exception and boxes
        assert all("\n" not in b for b in boxes) and any("<br><br># عنوان" in b for b in boxes)

    def test_comparison_view_shows_three_systems(self, monkeypatch):
        monkeypatch.setenv("PII_APP_ENGINE", "mock")
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.button(key="cmp_btn").click().run()
        assert not at.exception
        titles = [s.value for s in at.subheader]
        assert {"قبل التدريب", "بعد التدريب", "النظام الكامل"} <= set(titles)

    def test_unreachable_model_blocks_output_without_crashing(self, monkeypatch):
        monkeypatch.delenv("PII_APP_ENGINE", raising=False)
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.sidebar.text_input(key="host").set_value("http://127.0.0.1:9").run()
        at.button(key="run_btn").click().run()
        assert not at.exception and len(at.error) >= 1
        md = markdown_of(at)
        assert "حُجب النص" in md and "[هوية]" in md   # regex-only preview, flagged for review
