import json
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from app.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import vector_store
from app.services.llm_service import LLMService
from app.models.db_models import QueryLog
from app.models.schemas import RAGQueryResponse, RetrievedChunk, Citation


class RAGService:
    """Orchestrates chunk retrieval, metadata-filtered queries, grounded answer generation, and citation resolution."""

    @classmethod
    def query(
        cls,
        db: Session,
        question: str,
        top_k: int = 5,
        filter_year_min: Optional[int] = None,
        filter_year_max: Optional[int] = None,
        filter_sections: Optional[List[str]] = None,
        filter_paper_ids: Optional[List[str]] = None
    ) -> RAGQueryResponse:
        # 1. Embed query vector
        query_vector = EmbeddingService.embed_query(question)

        # 2. Retrieve top-k chunks from vector store
        raw_chunks = vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            filter_year_min=filter_year_min,
            filter_year_max=filter_year_max,
            filter_sections=filter_sections,
            filter_paper_ids=filter_paper_ids
        )

        retrieved_chunks = [
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

        # 3. Generate grounded answer with citations
        llm_result = LLMService.generate_answer(question, raw_chunks)
        answer_text = llm_result.get("answer", "")
        citations_data = llm_result.get("citations", [])

        citations = [
            Citation(
                paper_id=c.get("paper_id", ""),
                paper_title=c.get("paper_title", ""),
                year=c.get("year"),
                section_name=c.get("section_name", ""),
                chunk_id=c.get("chunk_id", 0),
                quote=c.get("quote", ""),
                evidence_text=c.get("evidence_text", ""),
                similarity_score=c.get("similarity_score")
            )
            for c in citations_data
        ]

        # 4. Save to QueryLog in SQLite
        try:
            log_entry = QueryLog(
                question=question,
                answer=answer_text,
                citations_json=json.dumps([c.dict() for c in citations]),
                retrieved_chunks_json=json.dumps([c.dict() for c in retrieved_chunks])
            )
            db.add(log_entry)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"[RAGService] Failed to log query: {e}")

        return RAGQueryResponse(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            grounded=len(citations) > 0,
            context_used=len(retrieved_chunks),
            llm_provider=llm_result.get("llm_provider"),
            llm_model=llm_result.get("llm_model"),
            llm_prompt_preview=llm_result.get("prompt_preview")
        )
