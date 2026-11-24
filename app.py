"""
app.py — Streamlit demo for the Research Paper QA (RAG) system.

Usage:
    streamlit run app.py
"""
import json
from pathlib import Path

import streamlit as st

from rag_core import PaperRAG, PAPERS, ROLE_ACCESS

st.set_page_config(page_title="Research Paper QA", layout="wide")

DEMO_FILE = Path(__file__).parent / "demo_questions.json"
DEMOS = json.loads(DEMO_FILE.read_text(encoding="utf-8")) if DEMO_FILE.exists() else []
ROLES = list(ROLE_ACCESS.keys())


@st.cache_resource(show_spinner="モデルとインデックスを読み込み中…")
def load_engine():
    return PaperRAG()


def use_demo(d):
    st.session_state["question"] = d["question"]
    st.session_state["role"] = d.get("role", "researcher")
    st.session_state["compare"] = bool(d.get("compare", False))


def run_query(question, role):
    """Retrieve, then stream the answer into the page."""
    import time
    with st.spinner("検索中…"):
        prep = engine.prepare(question, role)
    tm = prep["timings"]
    if prep["prompt"] is None:
        st.warning("この権限で参照できる文書の中に、該当する情報がありません。（LLMには渡していません）")
        st.session_state["audit_log"].append(prep["log"])
        return
    t = time.time()
    st.write_stream(engine.stream_answer(prep["prompt"]))
    t_gen = time.time() - t
    st.caption(
        f"検索 {tm.get('retrieval_total', 0):.1f}秒"
        f"（Embedding {tm.get('embed', 0):.1f} / BM25 {tm.get('bm25', 0):.2f} / "
        f"ベクトル {tm.get('vector', 0):.2f} / Rerank {tm.get('rerank', 0):.1f}）"
        f" ・ 回答生成 {t_gen:.1f}秒"
    )
    st.write("参照元")
    for s_ in prep["sources"]:
        lock = "（未公開）" if s_["access_level"] == "internal" else ""
        with st.expander(f"{lock}[{s_['label']}] {s_['title'][:70]}  (rerank {s_['score']:.2f})"):
            st.write(" ".join(s_["texts"]))
    st.session_state["audit_log"].append(prep["log"])


# ── state ───────────────────────────────────────────────────
st.session_state.setdefault("question", "")
st.session_state.setdefault("audit_log", [])

engine = load_engine()

# ── sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.header("設定")
    role = st.radio("ユーザー権限", ROLES, key="role",
                    format_func=lambda r: {"external_viewer": "外部ユーザー（公開論文のみ）",
                                           "researcher": "研究者（未公開論文も可）"}[r])
    compare = st.checkbox("2つの権限を並べて比較する", key="compare")
    st.divider()
    st.write("LLM：Ollama")
    st.write(f"Device：{engine.device}")
    st.write("Embedding：BGE-M3")
    st.write("Reranker：bge-reranker-v2-m3")
    st.divider()
    st.write("文書")
    for p in PAPERS:
        tag = "（未公開）" if p["access_level"] == "internal" else ""
        st.write(f"{p['doc_id']}  {p['year']}{tag}")
    st.divider()
    st.caption("処理の流れ：PDF解析 → Parent-Child分割 → ハイブリッド検索(BM25+ベクトル+RRF) "
               "→ リランキング → 権限フィルタ → 引用付き回答 → 監査ログ")

# ── main ────────────────────────────────────────────────────
st.title("Research Paper QA")
st.write("自分の研究論文を対象にした、ローカル環境で動くRAGシステムです。回答と一緒に、参照した論文名とページを表示します。")

if DEMOS:
    with st.container(border=True):
        st.write("デモ用の質問")
        c1, c2 = st.columns([4, 1])
        idx = c1.selectbox(
            "デモ質問", range(len(DEMOS)), label_visibility="collapsed",
            format_func=lambda i: f"{DEMOS[i]['step']}. {DEMOS[i]['category']}｜{DEMOS[i]['question'][:40]}",
        )
        c2.button("セット", on_click=use_demo, args=(DEMOS[idx],), use_container_width=True)
        st.caption(f"{DEMOS[idx]['point']}"
                   + ("（比較モードが自動でONになります）" if DEMOS[idx].get("compare") else ""))

question = st.text_area("質問（日本語・英語・中国語）", key="question", height=80)
run = st.button("質問する", type="primary")

if run and question.strip():
    tab_answer, tab_log = st.tabs(["回答", "監査ログ"])
    with tab_answer:
        if compare:
            left, right = st.columns(2)
            for col, r in zip((left, right), ROLE_ACCESS.keys()):
                with col:
                    st.subheader("外部ユーザー" if r == "external_viewer" else "研究者")
                    n_pub = sum(p["access_level"] == "public" for p in PAPERS)
                    n_int = sum(p["access_level"] == "internal" for p in PAPERS)
                    if r == "external_viewer":
                        st.caption(f"参照できる文書：公開論文 {n_pub}本（未公開論文は対象外）")
                    else:
                        st.caption(f"参照できる文書：公開論文 {n_pub}本 ＋ 未公開論文 {n_int}本")
                    run_query(question, r)
        else:
            run_query(question, role)
    with tab_log:
        st.dataframe(list(reversed(st.session_state["audit_log"])), use_container_width=True)
elif st.session_state["audit_log"]:
    with st.expander("監査ログ"):
        st.dataframe(list(reversed(st.session_state["audit_log"])), use_container_width=True)
