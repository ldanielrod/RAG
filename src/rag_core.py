import os
import re
from typing import List, Optional, Dict, Any, Tuple
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# -----------------------------
# Config (unificado)
# -----------------------------
CHROMA_DIR = os.getenv("RAG_CHROMA_DIR", "storage/chroma")
COLLECTION = os.getenv("RAG_COLLECTION", "policies_pdfs")
PROMPT_PATH = os.getenv("RAG_PROMPT_PATH", os.path.join("prompts", "system_prompt_citas.txt"))

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "qwen2.5-coder-3b-instruct-mlx")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")

TOP_K = int(os.getenv("RAG_TOP_K", "8"))
MIN_CONTEXT_CHARS = int(os.getenv("RAG_MIN_CONTEXT_CHARS", "50"))
DEBUG = os.getenv("RAG_DEBUG", "0") == "1"


# -----------------------------
# Helpers
# -----------------------------
def load_system_prompt(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No encontré el prompt en: {path}. "
            f"Crea el archivo o ajusta RAG_PROMPT_PATH en tu .env"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def doc_label(doc) -> str:
    src = doc.metadata.get("source_file") or os.path.basename(doc.metadata.get("source", "unknown"))
    page = doc.metadata.get("page", None)  # 0-based
    if page is not None:
        return f"{src} (pág. {page + 1})"
    return f"{src}"


def build_evidence_context(docs):
    """
    Contexto con IDs: [D1 | archivo (pág. X)] ...
    mapping: D# -> metadata (file/page/doc_id)
    """
    mapping: Dict[str, Dict[str, Any]] = {}
    lines: List[str] = []

    for i, d in enumerate(docs, start=1):
        did = f"D{i}"
        file_name = d.metadata.get("source_file") or os.path.basename(d.metadata.get("source", "unknown"))
        page0 = d.metadata.get("page", None)
        doc_id = d.metadata.get("doc_id", None)

        mapping[did] = {
            "file": file_name,
            "page": (page0 + 1) if page0 is not None else None,  # 1-based
            "doc_id": doc_id,
        }

        content = (d.page_content or "").strip()
        lines.append(f"[{did} | {doc_label(d)}]\n{content}")

    return "\n\n".join(lines), mapping


def extract_used_doc_ids(text: str) -> List[str]:
    ids = re.findall(r"\[(D\d+)\]", text)
    seen = set()
    ordered: List[str] = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            ordered.append(x)
    return ordered


def build_prompt(system_prompt: str, evidence_context: str, question: str) -> str:
    return f"""{system_prompt}

Contexto (Evidencias):
{evidence_context}

Pregunta:
{question}
"""


def dedup_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dedup por (file, page, doc_id) manteniendo orden.
    citations aquí son dicts para mantener el core desacoplado de FastAPI models.
    """
    seen: set[Tuple[Optional[str], Optional[int], Optional[str]]] = set()
    out: List[Dict[str, Any]] = []
    for c in citations:
        key = (c.get("file"), c.get("page"), c.get("doc_id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# -----------------------------
# RAG Engine (singleton)
# -----------------------------
class RAGEngine:
    def __init__(self):
        self.system_prompt = load_system_prompt(PROMPT_PATH)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings,
            collection_name=COLLECTION
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": TOP_K})

        self.llm = ChatOpenAI(
            base_url=LMSTUDIO_BASE_URL,
            api_key=LMSTUDIO_API_KEY,
            model=LMSTUDIO_MODEL,
            temperature=0
        )

        if DEBUG:
            print(f"[RAGEngine] base_url={LMSTUDIO_BASE_URL} model={LMSTUDIO_MODEL} top_k={TOP_K}")
            print(f"[RAGEngine] chroma_dir={CHROMA_DIR} collection={COLLECTION}")
            print(f"[RAGEngine] prompt_path={PROMPT_PATH}")

    def ask(self, question: str) -> Dict[str, Any]:
        docs = self.retriever.invoke(question)
        joined = " ".join((d.page_content or "") for d in docs).strip()

        if not docs or len(joined) < MIN_CONTEXT_CHARS:
            return {"answer": "No lo encontré en los PDFs indexados.", "citations": [], "retrieved_docs": len(docs)}

        evidence_context, mapping = build_evidence_context(docs)
        prompt = build_prompt(self.system_prompt, evidence_context, question)

        raw = self.llm.invoke(prompt).content.strip()

        used = extract_used_doc_ids(raw)
        citations: List[Dict[str, Any]] = []

        if used:
            for did in used:
                m = mapping.get(did)
                if m:
                    citations.append({"file": m["file"], "page": m["page"], "doc_id": m["doc_id"]})
        else:
            # fallback: top 3
            for did in list(mapping.keys())[:3]:
                m = mapping[did]
                citations.append({"file": m["file"], "page": m["page"], "doc_id": m["doc_id"]})
            raw = raw + "\n\n[Nota: el modelo no incluyó citas [D#].]"

        citations = dedup_citations(citations)
        return {"answer": raw, "citations": citations, "retrieved_docs": len(docs)}