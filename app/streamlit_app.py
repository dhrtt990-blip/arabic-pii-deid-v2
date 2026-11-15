"""Streamlit demo for Arabic PII de-identification. Runs on the team's own machine.

    streamlit run app/streamlit_app.py

Set PII_APP_ENGINE=mock to open the app in the rule-based dry-run mode (no
model needed). The default engine is the tuned model served by Ollama.
"""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.common import PLACEHOLDERS, redact, resolve_overlaps  # noqa: E402
from src.pipeline import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_HOST,
    DEFAULT_TUNED_MODEL,
    STATUS_EMPTY,
    STATUS_OK,
    STATUS_REPAIRED,
    DeidPipeline,
    MockClient,
    OllamaClient,
    run_model,
)

ENGINE_OLLAMA = "النموذج المحلي عبر Ollama"
ENGINE_MOCK = "وضع تجريبي بدون نموذج (Mock)"
SOURCE_LABELS = {"model": "النموذج", "regex": "القواعد (Regex)", "repair": "فاحص التسرّب"}
TYPE_LABELS = {
    "PERS": "اسم شخص", "LOC": "موقع", "ORG": "جهة", "DATE": "تاريخ",
    "NATIONAL_ID": "هوية / إقامة", "PHONE": "جوال", "MRN": "رقم ملف طبي", "EMAIL": "بريد إلكتروني",
}
EXAMPLE_TEXT = (
    "راجعت المريضة منال الرويلي عيادة السكري في مركز الريان الطبي بتاريخ 03/11/2025، "
    "رقم الهوية 1098765432 والجوال 0556677889."
)

RTL_CSS = """
<style>
.stApp, [data-testid="stSidebar"], [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {
    direction: rtl; text-align: right;
}
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp li, .stApp label { direction: rtl; text-align: right; }
textarea, input[type="text"] { direction: rtl; text-align: right; }
[data-testid="stSidebar"] input[type="text"], .ltr, .stApp pre, .stApp code {
    direction: ltr; text-align: left; unicode-bidi: embed;
}
.stApp ul { padding-right: 1.2rem; padding-left: 0; }
.box {
    border: 1px solid rgba(128,128,128,.45); border-radius: 8px; padding: 12px 14px;
    line-height: 2.1; white-space: pre-wrap; unicode-bidi: plaintext; text-align: start; min-height: 3rem;
}
.box.blocked { border-color: #c0392b; }
mark { border-radius: 4px; padding: 1px 4px; color: #111; }
mark.model { background: #ffe08a; }
mark.regex { background: #a8d8ff; }
mark.repair { background: #ffc2c2; }
mark sub { font-size: .65em; margin-inline-start: 4px; opacity: .75; }
table.ents { width: 100%; border-collapse: collapse; direction: rtl; }
table.ents th, table.ents td { border-bottom: 1px solid rgba(128,128,128,.35); padding: 6px 8px; text-align: right; }
.legend span { margin-inline-end: 12px; }
.muted { opacity: .7; }
</style>
"""


def esc(text: str) -> str:
    """Escape user-supplied text before it is placed inside HTML.

    Line breaks become <br>: a blank line inside an HTML block would end the
    block in Markdown, and the rest of the user's text would then be rendered
    as Markdown (headings, links) instead of plain text.
    """
    return html.escape(text, quote=True).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def highlight_html(text: str, entities: list[dict]) -> str:
    parts: list[str] = []
    pos = 0
    for e in sorted(entities, key=lambda e: e["start"]):
        if e["start"] < pos:
            continue
        css = e["source"] if e["source"] in SOURCE_LABELS else "model"
        label = TYPE_LABELS.get(e["type"], e["type"])
        parts.append(esc(text[pos:e["start"]]))
        parts.append(
            f'<mark class="{css}" title="{esc(label)} · {esc(SOURCE_LABELS[css])}">'
            f'{esc(text[e["start"]:e["end"]])}<sub>{esc(label)}</sub></mark>'
        )
        pos = e["end"]
    parts.append(esc(text[pos:]))
    return f'<div class="box">{"".join(parts)}</div>'


def entities_table_html(entities: list[dict]) -> str:
    if not entities:
        return '<p class="muted">لم يُكتشف أي مقطع حساس.</p>'
    rows = "".join(
        f"<tr><td>{esc(e['text'])}</td><td>{esc(TYPE_LABELS.get(e['type'], e['type']))}</td>"
        f"<td>{esc(SOURCE_LABELS.get(e['source'], e['source']))}</td><td>{esc(PLACEHOLDERS.get(e['type'], ''))}</td></tr>"
        for e in entities
    )
    return (
        '<table class="ents"><thead><tr><th>المقطع</th><th>النوع</th><th>المصدر</th><th>البديل</th></tr></thead>'
        f"<tbody>{rows}</tbody></table>"
    )


def text_box(text: str, blocked: bool = False) -> str:
    return f'<div class="box{" blocked" if blocked else ""}">{esc(text)}</div>'


LEGEND = (
    '<p class="legend"><span><mark class="model">النموذج</mark></span>'
    '<span><mark class="regex">القواعد (Regex)</mark></span>'
    '<span><mark class="repair">فاحص التسرّب</mark></span></p>'
)


def make_clients(engine: str, host: str, base_model: str, tuned_model: str):
    if engine == ENGINE_MOCK:
        return MockClient("weak"), MockClient("strong")
    return OllamaClient(base_model, host=host, few_shot=True), OllamaClient(tuned_model, host=host)


def model_only(client, text: str) -> dict:
    """The model with no regex layer and no leakage check (for the comparison view)."""
    ext = run_model(client, text)
    if not ext.ok:
        return {"ok": False, "error": ext.error, "entities": [], "redacted": None}
    spans = resolve_overlaps(ext.spans)
    return {"ok": True, "error": None, "entities": [s.to_dict(text) for s in spans], "redacted": redact(text, spans)}


def render_result(res: dict) -> None:
    status = res["status"]
    if status == STATUS_OK:
        st.success(res["message"])
    elif status == STATUS_REPAIRED:
        st.warning(res["message"])
    elif status == STATUS_EMPTY:
        st.info(res["message"])
        return
    else:
        st.error(res["message"])
        if res["model"].get("error"):
            st.markdown(f'<p class="ltr muted">{esc(res["model"]["error"])}</p>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.subheader("النص الأصلي")
        st.markdown(highlight_html(res["original"], res["entities"]) + LEGEND, unsafe_allow_html=True)
    with right:
        st.subheader("النص بعد الإخفاء")
        if res["released"]:
            st.markdown(text_box(res["redacted"]), unsafe_allow_html=True)
        else:
            st.markdown(text_box("حُجب النص ولم يُصرَّح بنشره.", blocked=True), unsafe_allow_html=True)
            if res.get("fallback_preview"):
                st.caption("معاينة غير مصرَّح بنشرها، وتحتاج مراجعة بشرية:")
                st.markdown(text_box(res["fallback_preview"], blocked=True), unsafe_allow_html=True)

    st.subheader("المقاطع المكتشفة ومصدرها")
    st.markdown(entities_table_html(res["entities"]), unsafe_allow_html=True)

    st.subheader("فاحص التسرّب")
    if res["leak_findings"]:
        st.error(f"رُصد {len(res['leak_findings'])} مقطع متبقٍ بعد الإخفاء.")
        for f in res["leak_findings"]:
            st.markdown(f"- {esc(f.get('type') or 'نص مكرر')}: {esc(f['text'])}", unsafe_allow_html=True)
    elif res["released"]:
        st.markdown("سليم: لا نمط حساس ولا مقطع مُخفى ظهر في النص الناتج.")
    else:
        st.markdown("لم يُنشر نص، فلا شيء للفحص.")

    if res["model"].get("not_in_text"):
        st.caption(f"تجاهل النظام {len(res['model']['not_in_text'])} مقطعاً ذكره النموذج ولا وجود له في النص.")
    if res["model"].get("raw"):
        with st.expander("مخرجات النموذج الخام"):
            for raw in res["model"]["raw"]:
                st.code(raw, language="json")


def render_panel(title: str, caption: str, panel: dict) -> None:
    st.subheader(title)
    st.caption(caption)
    if panel.get("status"):  # full pipeline result
        entities, redacted, released = panel["entities"], panel["redacted"], panel["released"]
        st.markdown(f"**الحالة:** {esc(panel['message'])}", unsafe_allow_html=True)
    else:
        entities, redacted, released = panel["entities"], panel["redacted"], panel["ok"]
        if not panel["ok"]:
            st.markdown(f'**الحالة:** فشل النموذج. <span class="ltr muted">{esc(panel["error"] or "")}</span>', unsafe_allow_html=True)
    st.markdown(f"عدد المقاطع المكتشفة: **{len(entities)}**")
    st.markdown(text_box(redacted) if released and redacted is not None else text_box("لا يوجد نص منشور.", blocked=True), unsafe_allow_html=True)
    st.markdown(entities_table_html(entities), unsafe_allow_html=True)


st.set_page_config(page_title="إخفاء المعلومات الشخصية", layout="wide")
st.markdown(RTL_CSS, unsafe_allow_html=True)
st.title("إخفاء المعلومات الشخصية من النص العربي")

with st.sidebar:
    st.header("الإعدادات")
    engines = [ENGINE_OLLAMA, ENGINE_MOCK]
    default_engine = ENGINE_MOCK if os.environ.get("PII_APP_ENGINE", "").lower() == "mock" else ENGINE_OLLAMA
    engine = st.radio("محرك الاستخراج", engines, index=engines.index(default_engine), key="engine")
    host = st.text_input("عنوان خادم Ollama", DEFAULT_HOST, key="host")
    base_model = st.text_input("النموذج قبل التدريب", DEFAULT_BASE_MODEL, key="base_model")
    tuned_model = st.text_input("النموذج بعد التدريب (LoRA)", DEFAULT_TUNED_MODEL, key="tuned_model")
    st.caption("يُرسل النص إلى خادم Ollama المحدد أعلاه فقط، ولا يُحفظ في أي ملف.")

base_client, tuned_client = make_clients(engine, host, base_model, tuned_model)

if engine == ENGINE_MOCK:
    st.warning("وضع تجريبي: المستخرِج هنا قواعد بسيطة وليس النموذج المدرَّب. يصلح لتجربة الواجهة فقط، لا للعرض النهائي ولا للتقييم.")
else:
    available, reason = tuned_client.is_available()
    if available:
        st.sidebar.success("النموذج المدرَّب جاهز.")
    else:
        st.sidebar.error("النموذج المدرَّب غير متاح.")
        st.sidebar.markdown(f'<p class="ltr muted">{esc(reason)}</p>', unsafe_allow_html=True)

tab_run, tab_compare = st.tabs(["إخفاء النص", "مقارنة قبل التدريب وبعده"])

with tab_run:
    text = st.text_area("النص المراد إخفاء معلوماته", value=EXAMPLE_TEXT, height=160, key="input_text")
    if st.button("إخفاء المعلومات", type="primary", key="run_btn"):
        with st.spinner("جارٍ المعالجة…"):
            st.session_state["result"] = DeidPipeline(tuned_client).run(text).to_dict()
    if st.session_state.get("result"):
        render_result(st.session_state["result"])

with tab_compare:
    st.markdown(
        "يعرض النص نفسه بثلاث طرق: النموذج قبل التدريب مع Prompt محكَم، النموذج بعد التدريب وحده، ثم النظام الكامل. "
        "العمودان الأولان بلا طبقة قواعد وبلا فاحص تسرّب."
    )
    cmp_text = st.text_area("النص", value=EXAMPLE_TEXT, height=140, key="cmp_text")
    if st.button("قارن", key="cmp_btn"):
        with st.spinner("جارٍ تشغيل الأنظمة الثلاثة…"):
            st.session_state["cmp"] = {
                "baseline": model_only(base_client, cmp_text),
                "tuned": model_only(tuned_client, cmp_text),
                "full": DeidPipeline(tuned_client).run(cmp_text).to_dict(),
                "names": (base_client.name, tuned_client.name),
            }
    cmp = st.session_state.get("cmp")
    if cmp:
        c1, c2, c3 = st.columns(3)
        with c1:
            render_panel("قبل التدريب", cmp["names"][0], cmp["baseline"])
        with c2:
            render_panel("بعد التدريب", cmp["names"][1], cmp["tuned"])
        with c3:
            render_panel("النظام الكامل", f"{cmp['names'][1]} + Regex + فاحص التسرّب", cmp["full"])
