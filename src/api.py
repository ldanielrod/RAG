from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.rag_core import RAGEngine, LMSTUDIO_MODEL

app = FastAPI(title="RAG PDFs API", version="0.1.0")
rag = RAGEngine()

class UserContext(BaseModel):
    oid: Optional[str] = None
    upn: Optional[str] = None
    tenant: Optional[str] = None

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    channel_type: str = Field("personal", description="personal | channel")
    user: Optional[UserContext] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None

class Citation(BaseModel):
    file: str
    page: Optional[int] = None
    doc_id: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    safe_to_share: bool = True
    meta: Dict[str, Any] = {}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    out = rag.ask(req.question)
    return AskResponse(
        answer=out["answer"],
        citations=[Citation(**c) for c in out["citations"]],
        safe_to_share=True,
        meta={
            "channel_type": req.channel_type,
            "retrieved_docs": out["retrieved_docs"],
            "model": LMSTUDIO_MODEL,
        },
    )