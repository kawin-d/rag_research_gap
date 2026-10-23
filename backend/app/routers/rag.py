import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.schemas import RAGQueryRequest, RAGQueryResponse, RetrievedChunk, LLMConfigRequest
from app.models.db_models import QueryLog
from app.services.rag_service import RAGService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import vector_store

router = APIRouter(prefix="/rag", tags=["Research Paper Analysis & RAG Layer"])


@router.post("/ask", response_model=RAGQueryResponse)
def ask_question(req: RAGQueryRequest, db: Session = Depends(get_db)):
    """Ask a natural language research question over the ingested paper corpus.
    
    Returns:
    - Answer synthesized strictly from retrieved text.
    - Source citations explicitly noting Paper Title, Year, and Section.
    - All retrieved text chunks for complete transparency.
    """
    clean_q = (req.question or "").strip()
    if not clean_q:
        raise HTTPException(
            status_code=400,
            detail="Please enter a research question. Empty or whitespace-only queries cannot be processed."
        )
    if len(clean_q) < 3:
        raise HTTPException(
            status_code=400,
            detail="Question is too brief. Please enter a more specific research query or topic."
        )

    if len(vector_store.metadata) == 0:
        raise HTTPException(
            status_code=400,
            detail="The knowledge base is empty. Please ingest papers via /api/papers/search-arxiv or /api/sample/load first."
        )

    response = RAGService.query(
        db=db,
        question=clean_q,
        top_k=req.top_k,
        filter_year_min=req.filter_year_min,
        filter_year_max=req.filter_year_max,
        filter_sections=req.filter_sections,
        filter_paper_ids=req.filter_paper_ids
    )

    return response


@router.post("/retrieve", response_model=List[RetrievedChunk])
def retrieve_chunks(req: RAGQueryRequest):
    """Retrieve raw relevant chunks without generating an answer (useful for inspecting retrieval and evaluating recall@k)."""
    if len(vector_store.metadata) == 0:
        return []

    q_vec = EmbeddingService.embed_query(req.question)
    raw_chunks = vector_store.search(
        query_vector=q_vec,
        top_k=req.top_k,
        filter_year_min=req.filter_year_min,
        filter_year_max=req.filter_year_max,
        filter_sections=req.filter_sections,
        filter_paper_ids=req.filter_paper_ids
    )

    return [
        RetrievedChunk(
            chunk_id=c.get("chunk_id", 0),
            paper_id=c.get("paper_id", ""),
            paper_title=c.get("paper_title", "Unknown"),
            year=c.get("year"),
            section_name=c.get("section_name", "general"),
            text=c.get("text", ""),
            similarity_score=round(c.get("similarity_score", 0.0), 4)
        )
        for c in raw_chunks
    ]


@router.get("/history")
def get_query_history(limit: int = 20, db: Session = Depends(get_db)):
    """Fetch previous question-answer logs and their citations."""
    logs = db.query(QueryLog).order_by(QueryLog.created_at.desc()).limit(limit).all()
    out = []
    for l in logs:
        out.append({
            "id": l.id,
            "question": l.question,
            "answer": l.answer,
            "citations": json.loads(l.citations_json) if l.citations_json else [],
            "retrieved_chunks": json.loads(l.retrieved_chunks_json) if l.retrieved_chunks_json else [],
            "created_at": l.created_at
        })
    return out


@router.get("/llm-status")
def get_llm_status():
    """Check the currently active LLM provider and status."""
    from app.services.llm_service import LLMService
    return LLMService.get_status()


@router.post("/llm-config")
def update_llm_config(cfg: LLMConfigRequest):
    """Dynamically update LLM provider, API key, and model at runtime."""
    from app.services.llm_service import LLMService
    return LLMService.update_config(
        provider=cfg.provider,
        api_key=cfg.api_key,
        model=cfg.model,
        base_url=cfg.base_url
    )


