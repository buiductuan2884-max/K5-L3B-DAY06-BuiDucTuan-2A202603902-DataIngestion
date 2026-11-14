"""Demo truy xuất: streamlit run app.py (tái dùng src/, không ingest lại)."""
import json
import sys
from pathlib import Path

import csv
import os
import re

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.embeddings import MyEmbedder  # noqa: E402
from src.evaluation import gold_pages, is_hit  # noqa: E402
from src.store import ChromaStore  # noqa: E402

R = HERE / "results"
st.set_page_config(page_title="Demo truy xuất", layout="wide")


@st.cache_resource
def get_store():
    return ChromaStore()


@st.cache_resource
def get_embedder():
    return MyEmbedder()


def embed_q(text: str):
    """Embed qua API; nếu API lỗi mà câu hỏi là câu benchmark thì dùng vector đã lưu (cùng model)."""
    try:
        return get_embedder().embed_query(text)
    except Exception:
        p = R / "query_vectors.json"
        cached = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        if text in cached:
            st.warning("OpenAI không phản hồi, dùng vector đã lưu cho câu hỏi này.")
            return cached[text]
        raise


@st.cache_data
def load_json(p: str):
    return json.loads(Path(p).read_text(encoding="utf-8"))


@st.cache_data
def load_jsonl(p: str):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x]


@st.cache_data
def load_csv(p: str):
    return list(csv.DictReader(open(p, encoding="utf-8-sig", newline="")))


def table(rows: list[dict]):
    """Bảng markdown: st.dataframe/table cần pyarrow, DLL bị Application Control chặn trên máy này."""
    cols = list(rows[0])
    st.markdown("\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                          ["| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |" for r in rows]))


def hist(lens: list[int]):
    bins = {}
    for n in lens:
        bins[n // 50 * 50] = bins.get(n // 50 * 50, 0) + 1
    vals = [{"chars": str(k), "count": v} for k, v in sorted(bins.items())]
    st.vega_lite_chart({"data": {"values": vals}, "mark": "bar", "height": 160,
                        "encoding": {"x": {"field": "chars", "type": "ordinal", "sort": None},
                                     "y": {"field": "count", "type": "quantitative"}}}, width="stretch")


def cards(hits, q=None, key=""):
    """hits: [{rank?, score, page, page_end, chapter, section, text, chunk_id}]; q: câu benchmark để đánh dấu."""
    for i, h in enumerate(hits, 1):
        mark = "" if q is None else (" ✓" if h["hit"] else " ✗")
        with st.container(border=True):
            st.markdown(f"**#{i}{mark}** · điểm {h['score']:.3f} · tr. {h['page']}-{h['page_end']} · "
                        f"chương {h['chapter']} · mục {h['section']}")
            st.text(h["text"][:300] + ("…" if len(h["text"]) > 300 else ""))
            if len(h["text"]) > 300:
                with st.expander("Toàn văn"):
                    st.text(h["text"])


def from_store(res, q):
    return [{"score": r["score"], "page": r["metadata"]["page"], "page_end": r["metadata"]["page_end"],
             "chapter": r["metadata"]["chapter"], "section": r["metadata"]["section"], "text": r["text"],
             "hit": is_hit(r, q) if q else False} for r in res]


questions = load_json(str(HERE.parent / "benchmark" / "questions.json"))
qmap = {q["id"]: q for q in questions}
tab1, tab2, tab3, tab4 = st.tabs(["Kho tri thức", "Hỏi thử", "Benchmark", "Trả lời (LLM)"])

with tab1:
    man = load_json(str(HERE.parent / "data" / "manifest.json"))
    for s in man["sources"]:
        st.subheader(s["title"])
        st.write(f"Nguồn: {s['source_path_or_url']}  \nGiấy phép: {s['license_or_permission']}")
        table([{"chương": k, "trang tính": f"{v[0]}-{v[1]} ({v[1]-v[0]+1})"} for k, v in s["counted_pages"].items()])
    st.write("Trường lọc:", man["filter_field"])
    stats = load_json(str(R / "chunk_stats.json"))
    table([{"chiến lược": k, **v} for k, v in stats.items()])
    chunks = {s: load_jsonl(str(R / f"chunks_{s}.jsonl")) for s in "AB"}
    for s in "AB":
        st.caption(f"Phân bố độ dài chunk {s} (số ký tự, bin 50)")
        hist([len(c["text"]) for c in chunks[s]])
    st.divider()
    pages = load_json(str(R / "pages_normalized.json"))
    pmap = {p["page"]: p for p in pages}
    pn = st.selectbox("Chọn trang", sorted(pmap))
    left, right = st.columns(2)
    left.text(pmap[pn]["text"])
    for s in "AB":
        right.markdown(f"**Chunk {s}**")
        for c in chunks[s]:
            if c["page"] <= pn <= c["metadata"]["page_end"]:
                right.caption(f"{c['chunk_id']} · mục {c['metadata']['section']} · {len(c['text'])} ký tự")

with tab2:
    def pick():  # chọn câu mẫu -> điền câu hỏi + bộ lọc
        q = qmap.get(st.session_state.ex)
        if q:
            st.session_state.qtext = q["question"]
            st.session_state.chap = (q["filter"] or {}).get("chapter", "Không lọc")

    st.selectbox("Câu hỏi mẫu", [""] + list(qmap), key="ex", on_change=pick,
                 format_func=lambda i: "-" if not i else f"{i}: {qmap[i]['question']}")
    question = st.text_input("Câu hỏi", key="qtext")
    chap = st.selectbox("Lọc chương", ["Không lọc", 3, 4, 5], key="chap")
    bq = qmap.get(st.session_state.get("ex"))
    if bq and bq["question"] != question:
        bq = None  # đã sửa câu hỏi -> không còn khớp gold
    if st.button("Tìm"):
        if not question.strip():
            st.error("Nhập câu hỏi trước.")
        else:
            try:
                vec = embed_q(question)  # embed 1 lần
                store = get_store()
                where = None if chap == "Không lọc" else {"chapter": chap}
                runs = {"A": store.search("A", vec), "B": store.search("B", vec),
                        "C": store.search("B", vec, where=where)}
                st.session_state.res = {k: from_store(v, bq) for k, v in runs.items()}
                st.session_state.res_where, st.session_state.res_q = where, bq
            except Exception as e:  # lỗi API/Chroma: không giả vector
                st.error(f"Lỗi: {e}")
    if "res" in st.session_state:
        titles = {"A": "A (strategy_A)", "B": "B (strategy_B)", "C": "C (B + lọc chương)"}
        for col, k in zip(st.columns(3), "ABC"):
            with col:
                st.subheader(titles[k])
                if k == "C" and not st.session_state.res_where:
                    st.caption("C = B không lọc")
                cards(st.session_state.res[k], st.session_state.res_q)

with tab3:
    table(load_csv(str(R / "metrics.csv")))
    st.caption("Tuân thủ bộ lọc")
    table(load_csv(str(R / "filter_compliance.csv")))
    traces = {s: {t["id"]: t for t in load_jsonl(str(R / f"traces_{s}.jsonl"))} for s in "ABC"}
    qid = st.selectbox("Câu hỏi", list(qmap), format_func=lambda i: f"{i}: {qmap[i]['question']}")
    q = qmap[qid]
    st.write(f"split: {q['split']} · loại: {q['type']} · lọc: {q['filter']}")
    st.write(f"**Câu hỏi:** {q['question']}")
    st.write(f"**Đáp án:** {q['expected_answer']}")
    st.write(f"**Trang gold:** {gold_pages(q)}")
    st.write(f"**Trích dẫn:** {q['evidence_quote']}")
    for col, k in zip(st.columns(3), "ABC"):
        t = traces[k][qid]
        with col:
            st.subheader(k)
            if t["evidence_split_flag"]:
                st.warning("evidence split")
            cards([{**h, "hit": h["is_hit"]} for h in t["top3"]], q)

with tab4:
    cm = os.getenv("CHAT_MODEL") or "gpt-4o-mini"
    PRICE_IN, PRICE_OUT = 0.15, 0.60  # giả định giá gpt-4o-mini, USD / 1M token (input / output); chỉ là ước tính

    def chat(system, user):
        r = get_embedder().client.chat.completions.create(
            model=cm, temperature=0, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content.strip(), r.usage

    q4 = st.text_area("Câu hỏi", key="q4")
    ch4 = st.selectbox("Lọc chương", ["Không lọc", 3, 4, 5], key="ch4")
    strat = st.radio("Chiến lược", ["B", "C"], horizontal=True, key="strat4")
    tr4 = st.checkbox("Dịch câu hỏi sang tiếng Anh trước khi tìm", key="tr4")
    if strat == "C" and ch4 == "Không lọc":
        st.caption("C không lọc = B")
    if st.button("Trả lời", key="go4"):
        if not q4.strip():
            st.error("Nhập câu hỏi trước.")
        else:
            try:
                used, query = [], q4
                if tr4:
                    query, u = chat("Translate the user's question to English. Output only the translation. "
                                    "If it is already English, return it unchanged.", q4)
                    used.append(u)
                    st.info(f"Câu hỏi đã dịch: {query}")
                where = {"chapter": ch4} if strat == "C" and ch4 != "Không lọc" else None
                hits = from_store(get_store().search("B", get_embedder().embed_query(query), where=where), None)
                nl = chr(10)
                ctx = (nl * 2).join(f"[{i}] (page {h['page']}, chapter {h['chapter']}, section {h['section']}){nl}{h['text']}"
                                    for i, h in enumerate(hits, 1))
                # chọn ngôn ngữ bằng code, không để model tự đoán (gpt-4o-mini từng trả lời tiếng Bồ Đào Nha)
                lang = "Vietnamese" if re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
                                                 q4.lower()) else "English"
                ans, u = chat(f"Answer ONLY from the provided excerpts. Write the whole answer in {lang}. Plain text, no LaTeX. "
                              "Cite each claim as [p.<page>] using the page numbers given. If the excerpts do "
                              f"not contain the answer, say so plainly in {lang} and do not use outside "
                              "knowledge.", f"Question: {q4}{nl * 2}Excerpts:{nl}{ctx}")
                used.append(u)
                pt, ct = sum(x.prompt_tokens for x in used), sum(x.completion_tokens for x in used)
                st.markdown("### Trả lời")
                st.write(ans)
                st.caption(f"Token: prompt {pt}, completion {ct} · chi phí ước tính ≈ "
                           f"${(pt * PRICE_IN + ct * PRICE_OUT) / 1e6:.5f} (giả định giá {cm}: $0.15/1M vào, $0.60/1M ra)")
                for i, h in enumerate(hits, 1):
                    with st.expander(f"#{i} · điểm {h['score']:.3f} · tr. {h['page']} · chương {h['chapter']} · mục {h['section']}"):
                        st.text(h["text"])
            except Exception as e:  # lỗi API/mạng: không giả đầu ra
                st.error(f"Lỗi: {e}")
