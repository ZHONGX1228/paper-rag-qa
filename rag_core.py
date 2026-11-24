"""
rag_core.py — Research Paper QA (RAG) core logic
Refactored from Paper_RAG_PoC.ipynb (Phase 1-8) for the Streamlit demo app.
"""
import os
import re
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np

# Use locally cached models only — no network check at startup (works offline).
# First-time download: run with PAPER_RAG_ONLINE=1
if os.environ.get("PAPER_RAG_ONLINE") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("PAPER_RAG_DATA", "/Users/astro/Desktop/paper_rag/data")
INDEX_DIR = os.environ.get("PAPER_RAG_INDEX", str(Path(__file__).parent / "index"))

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
PREFERRED_MODELS = ["qwen2.5:7b", "llama3.2:3b"]   # Qwen handles Japanese better

PAPERS = [
    {"doc_id": "JAF2019", "file": "论文JAF.pdf",
     "title": "Pricing Liquidity Risk on the Tokyo Stock Exchange: Empirical Analysis Using Multiple Liquidity Measures",
     "journal": "Journal of Accounting and Finance", "year": 2019, "access_level": "public"},
    {"doc_id": "JCR2020", "file": "论文JCR.pdf",
     "title": "Stock Liquidity Premium on the Tokyo Stock Exchange",
     "journal": "Journal of Critical Reviews", "year": 2020, "access_level": "public"},
    {"doc_id": "MF2021", "file": "MF_published.pdf",
     "title": "Performances of Liquidity Factors in the Stock Market Cycle: Evidence from the Tokyo Stock Exchange",
     "journal": "Managerial Finance", "year": 2021, "access_level": "public"},
    {"doc_id": "DP2021", "file": "Discussion Paper.pdf",
     "title": "Neural Network Application in Predicting Stock Returns: Evidence from Japan",
     "journal": "Shiga University Discussion Paper E-8", "year": 2021, "access_level": "public"},
    {"doc_id": "WP_CN", "file": "Main.pdf",
     "title": "Multifractal Analysis of the Relationship Between Information Asymmetry and Liquidity in China",
     "journal": "Working paper (unpublished)", "year": 2023, "access_level": "internal"},
]

ROLE_ACCESS = {
    "external_viewer": ["public"],               # e.g. client / interviewer
    "researcher":      ["public", "internal"],   # author / research team
}

PROMPT = """You are a research assistant for the papers of Xin Zhong.
Answer the question using ONLY the sources below.
- Cite every claim with its source label, e.g. [MF2021 p.7].
- If the sources do not contain the answer, say so. Do not use outside knowledge.
- Answer in the same language as the question, in 3-5 sentences.
- If the question is in Japanese, answer in natural Japanese only (no Chinese words).

Sources:
{context}

Question: {question}
Answer:"""

# ─────────────────────────────────────────────────────────────
# Phase 1: Column-aware PDF text extraction
# ─────────────────────────────────────────────────────────────
REF_HEADING = re.compile(r"^\s*(references|bibliography|参考文献)\s*$", re.I)
NOISE_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"emerald\.com|insight/0307|© ?\d{4}|creativecommons|doi\.org|DOI:", re.I),
    re.compile(r"^(Journal of Accounting and Finance|Journal of Critical Reviews|Managerial Finance)\b.*\d", re.I),
]


def _is_noise(line: str) -> bool:
    return any(p.search(line) for p in NOISE_PATTERNS)


def extract_pdf_pages(filepath: str) -> list:
    import fitz  # PyMuPDF
    doc = fitz.open(filepath)
    pages, in_refs = [], False
    for pno in range(len(doc)):
        page = doc[pno]
        mid = page.rect.width / 2
        blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]

        def col_of(b):
            if b[2] <= mid + 15:
                return 1
            if b[0] >= mid - 15:
                return 2
            return 0
        col_tops = [b[1] for b in blocks if col_of(b) != 0]
        first_col_y = min(col_tops) if col_tops else 0

        def order(b):
            c = col_of(b)
            if c == 0:
                return (0 if b[1] <= first_col_y else 3, b[1])
            return (c, b[1])

        lines_out = []
        for b in sorted(blocks, key=order):
            for line in b[4].split("\n"):
                line = re.sub(r"\s+", " ", line).strip()
                if not line:
                    continue
                if REF_HEADING.match(line):
                    in_refs = True
                    break
                if not _is_noise(line):
                    lines_out.append(line)
            if in_refs:
                break
        text = " ".join(lines_out)
        text = re.sub(r"(\w)- (\w)", r"\1\2", text)
        if len(text) > 200:
            pages.append({"page": pno + 1, "text": text})
        if in_refs:
            break
    return pages

# ─────────────────────────────────────────────────────────────
# Phase 2: Parent-child chunking + metadata
# ─────────────────────────────────────────────────────────────
SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+(?=[A-Z0-9(\"“])")


def split_sentences(text: str) -> list:
    return [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) > 20]


def make_chunk_id(doc_id: str, page: int, index: int) -> str:
    raw = f"{doc_id}::page{page}::{index}"
    return f"{doc_id}__p{page}__{index:04d}__{hashlib.md5(raw.encode()).hexdigest()[:8]}"


def build_chunks(pages, paper, parent_store, child_chunks, child_words=70, parent_children=4):
    for pg in pages:
        sents = split_sentences(pg["text"])
        windows, buf = [], []
        for s in sents:
            buf.append(s)
            if sum(len(x.split()) for x in buf) >= child_words:
                windows.append(" ".join(buf))
                buf = []
        if buf:
            windows.append(" ".join(buf))
        for gi in range(0, len(windows), parent_children):
            group = windows[gi:gi + parent_children]
            parent_id = f"{paper['doc_id']}__p{pg['page']}__parent{gi // parent_children}"
            parent_store[parent_id] = {
                "parent_chunk_id": parent_id, "doc_id": paper["doc_id"],
                "title": paper["title"], "page": pg["page"],
                "access_level": paper["access_level"], "full_text": " ".join(group),
            }
            for j, w in enumerate(group):
                child_chunks.append({
                    "chunk_id": make_chunk_id(paper["doc_id"], pg["page"], gi + j),
                    "parent_chunk_id": parent_id, "doc_id": paper["doc_id"],
                    "title": paper["title"], "journal": paper["journal"],
                    "year": paper["year"], "page": pg["page"],
                    "access_level": paper["access_level"], "text": w,
                })

# ─────────────────────────────────────────────────────────────
# Index persistence (build once, load fast at demo time)
# ─────────────────────────────────────────────────────────────
def save_index(child_chunks, parent_store, vectors, index_dir=INDEX_DIR):
    d = Path(index_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "child_chunks.json").write_text(json.dumps(child_chunks, ensure_ascii=False), encoding="utf-8")
    (d / "parent_store.json").write_text(json.dumps(parent_store, ensure_ascii=False), encoding="utf-8")
    np.save(d / "vectors.npy", vectors.astype(np.float32))


def load_index(index_dir=INDEX_DIR):
    d = Path(index_dir)
    child_chunks = json.loads((d / "child_chunks.json").read_text(encoding="utf-8"))
    parent_store = json.loads((d / "parent_store.json").read_text(encoding="utf-8"))
    vectors = np.load(d / "vectors.npy")
    return child_chunks, parent_store, vectors

# ─────────────────────────────────────────────────────────────
# Phase 4 helpers: tokenizer + RRF
# ─────────────────────────────────────────────────────────────
TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
STOP = set("the a an of in and or to is are was were be by for on with as at from that this these those it its we our than which".split())


def tokenize(text: str) -> list:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def rrf_fusion(results_lists: list, k: int = 60) -> list:
    scores = {}
    for results in results_lists:
        for rank, cid in enumerate(results):
            scores[cid] = scores.get(cid, 0) + 1 / (rank + k)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def choose_model() -> str:
    import ollama
    try:
        resp = ollama.list()
        models = resp.models if hasattr(resp, "models") else resp.get("models", [])
        installed = {getattr(m, "model", None) or m.get("name") for m in models}
    except Exception:
        installed = set()
    for m in PREFERRED_MODELS:
        if m in installed:
            return m
    return PREFERRED_MODELS[-1]

# ─────────────────────────────────────────────────────────────
# Engine: Phase 3-7 at query time
# ─────────────────────────────────────────────────────────────
def pick_device() -> str:
    """Use Apple GPU (MPS) when available; override with PAPER_RAG_DEVICE=cpu."""
    env = os.environ.get("PAPER_RAG_DEVICE")
    if env:
        return env
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


RERANK_CANDIDATES = 10    # was 20 → halves reranker work
TOP_SOURCES = 4           # chunks passed to the LLM
LLM_OPTIONS = {"temperature": 0, "num_ctx": 4096}
KEEP_ALIVE = "60m"        # keep the LLM in memory between questions
MIN_RERANK = 0.10         # below this, treat as "no relevant document" (no LLM call)


class PaperRAG:
    def __init__(self, index_dir=INDEX_DIR):
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance, PointStruct
        from rank_bm25 import BM25Okapi

        self.device = pick_device()
        self.child_chunks, self.parent_store, vectors = load_index(index_dir)
        self.chunk_map = {c["chunk_id"]: c for c in self.child_chunks}

        # BM25 built ONCE per role (not on every question)
        corpus = [tokenize(c["text"]) for c in self.child_chunks]
        self.bm25 = {}
        for role, allowed in ROLE_ACCESS.items():
            idx = [i for i, c in enumerate(self.child_chunks) if c["access_level"] in allowed]
            self.bm25[role] = (idx, BM25Okapi([corpus[i] for i in idx]))

        self.embedding_model = SentenceTransformer(EMBED_MODEL, device=self.device)
        self.reranker = CrossEncoder(RERANK_MODEL, device=self.device, max_length=512)

        self.qdrant = QdrantClient(":memory:")
        self.qdrant.create_collection(
            collection_name="papers",
            vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
        )
        points = [PointStruct(id=i, vector=vectors[i].tolist(), payload=c)
                  for i, c in enumerate(self.child_chunks)]
        for s in range(0, len(points), 256):
            self.qdrant.upsert(collection_name="papers", points=points[s:s + 256])

        self.llm_model = choose_model()
        self.warmup()

    def warmup(self):
        """Run every model once at startup so the first real question is fast."""
        try:
            self.embedding_model.encode(["warm up"], normalize_embeddings=True)
            self.reranker.predict([["warm up", "warm up text"]] * 4)
        except Exception:
            pass
        try:
            import ollama
            ollama.generate(model=self.llm_model, prompt="hi", keep_alive=KEEP_ALIVE,
                            options={"num_predict": 1})
        except Exception:
            pass

    # Phase 4: hybrid search, access filter applied BEFORE retrieval
    def hybrid_search(self, query: str, role: str, top_k: int = RERANK_CANDIDATES, timings=None) -> list:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        timings = timings if timings is not None else {}
        allowed = ROLE_ACCESS.get(role, [])
        if not allowed:
            return []

        t = time.time()
        idx, bm25 = self.bm25[role]
        scores = bm25.get_scores(tokenize(query))
        bm25_ranked = [self.child_chunks[idx[j]]["chunk_id"]
                       for j in np.argsort(scores)[::-1][:20] if scores[j] > 0]
        timings["bm25"] = time.time() - t

        t = time.time()
        qv = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        timings["embed"] = time.time() - t

        t = time.time()
        hits = self.qdrant.query_points(
            collection_name="papers", query=qv.tolist(), limit=20,
            query_filter=Filter(must=[FieldCondition(key="access_level", match=MatchAny(any=allowed))]),
        ).points
        vector_ranked = [h.payload["chunk_id"] for h in hits]
        timings["vector"] = time.time() - t

        fused = rrf_fusion([bm25_ranked, vector_ranked])
        return [{"chunk": self.chunk_map[cid], "rrf_score": s} for cid, s in fused[:top_k]]

    # Phase 5: cross-encoder rerank
    def rerank(self, query: str, candidates: list, top_k: int = TOP_SOURCES, timings=None) -> list:
        timings = timings if timings is not None else {}
        if not candidates:
            return []
        t = time.time()
        scores = self.reranker.predict([[query, c["chunk"]["text"]] for c in candidates],
                                       batch_size=len(candidates))
        timings["rerank"] = time.time() - t
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [{**c, "rerank_score": float(s)} for s, c in ranked[:top_k]]

    # Retrieval + prompt building (no LLM yet)
    def prepare(self, question: str, role: str = "researcher") -> dict:
        timings = {}
        t0 = time.time()
        top = self.rerank(question, self.hybrid_search(question, role, timings=timings), timings=timings)
        timings["retrieval_total"] = time.time() - t0
        log = {"time": datetime.now().strftime("%H:%M:%S"), "role": role, "question": question}

        if not top or top[0]["rerank_score"] < MIN_RERANK:
            return {"prompt": None, "sources": [], "timings": timings,
                    "log": {**log, "status": "NOT_FOUND", "sources": []}}

        by_label, seen = {}, set()
        for r in top:
            pid = r["chunk"]["parent_chunk_id"]
            if pid in seen:
                continue
            seen.add(pid)
            par = self.parent_store[pid]
            label = f"{par['doc_id']} p.{par['page']}"
            if label not in by_label:
                by_label[label] = {"label": label, "doc_id": par["doc_id"], "title": par["title"],
                                   "access_level": par["access_level"],
                                   "score": r["rerank_score"], "texts": []}
            by_label[label]["texts"].append(par["full_text"])
        sources = list(by_label.values())
        blocks = [f"[{s['label']}] " + " ".join(s["texts"]) for s in sources]
        prompt = PROMPT.format(context="\n\n".join(blocks), question=question)
        return {"prompt": prompt, "sources": sources, "timings": timings,
                "log": {**log, "status": "ANSWERED", "sources": [s["label"] for s in sources]}}

    # Phase 7: streaming generation (text appears token by token)
    def stream_answer(self, prompt: str):
        import ollama
        try:
            for part in ollama.chat(model=self.llm_model,
                                    messages=[{"role": "user", "content": prompt}],
                                    options=LLM_OPTIONS, keep_alive=KEEP_ALIVE, stream=True):
                yield part["message"]["content"]
        except Exception as e:
            yield f"\n\n⚠️ LLM error: {e}\n(Ollamaアプリを起動してから再試行してください)"
