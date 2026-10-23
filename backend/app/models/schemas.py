from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ==========================================
# Paper Schemas
# ==========================================
class PaperBase(BaseModel):
    paper_id: str
    title: str
    authors: List[str] = []
    year: Optional[int] = None
    abstract: str = ""
    citation_count: int = 0
    url: str = ""


class PaperCreate(PaperBase):
    pdf_path: str = ""
    sections: Dict[str, str] = {}


class PaperResponse(PaperBase):
    id: int
    pdf_path: str = ""
    sections: Dict[str, str] = {}
    total_chunks: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


class ArxivSearchRequest(BaseModel):
    query: str = Field(..., description="Topic or query e.g. 'few-shot learning in NLP'")
    max_results: int = Field(10, ge=1, le=100, description="Number of papers to fetch (1-100)")
    download_pdfs: bool = Field(True, description="Whether to download and parse PDFs automatically")


class PaperListResponse(BaseModel):
    total: int
    papers: List[PaperResponse]


# ==========================================
# Chunk & RAG Schemas
# ==========================================
class ChunkResponse(BaseModel):
    id: int
    paper_uid: str
    paper_title: Optional[str] = None
    year: Optional[int] = None
    section_name: str
    text: str
    token_count: int


class RetrievedChunk(BaseModel):
    chunk_id: int
    paper_id: str
    paper_title: str
    year: Optional[int] = None
    section_name: str
    text: str
    similarity_score: float


class Citation(BaseModel):
    paper_id: str
    paper_title: str
    year: Optional[int] = None
    section_name: str
    chunk_id: int
    quote: str
    evidence_text: Optional[str] = ""
    similarity_score: Optional[float] = None


class RAGQueryRequest(BaseModel):
    question: str = Field(..., description="Research question to ask over the paper corpus")
    top_k: int = Field(5, ge=1, le=20, description="Number of relevant chunks to retrieve")
    filter_year_min: Optional[int] = Field(None, description="Optional minimum publication year filter")
    filter_year_max: Optional[int] = Field(None, description="Optional maximum publication year filter")
    filter_sections: Optional[List[str]] = Field(None, description="Filter by section e.g. ['methodology', 'results']")
    filter_paper_ids: Optional[List[str]] = Field(None, description="Filter by specific paper IDs")


class RAGQueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk]
    grounded: bool = True
    context_used: int = 0
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_prompt_preview: Optional[str] = None


class LLMConfigRequest(BaseModel):
    provider: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None



# ==========================================
# Research Gap Schemas
# ==========================================
class GapWeights(BaseModel):
    w_sparsity: float = Field(0.25, ge=0.0, le=1.0, description="Weight for semantic sparsity signal")
    w_decay: float = Field(0.25, ge=0.0, le=1.0, description="Weight for temporal decay signal")
    w_contradiction: float = Field(0.25, ge=0.0, le=1.0, description="Weight for contradiction density signal")
    w_admission: float = Field(0.25, ge=0.0, le=1.0, description="Weight for explicit author admission signal")


class GapAnalysisRequest(BaseModel):
    weights: Optional[GapWeights] = None
    num_clusters: Optional[int] = Field(None, ge=2, le=25, description="Number of clusters (auto-estimated if None)")


class EvidenceQuote(BaseModel):
    paper_id: str
    paper_title: str
    year: Optional[int] = None
    section_name: str
    quote: str
    signal_type: str  # 'admission', 'contradiction', 'sparsity'


class ClusterGapResponse(BaseModel):
    cluster_id: int
    cluster_label: int
    topic_title: str
    paper_ids: List[str]
    paper_titles: List[str]
    paper_count: int
    
    # 4 Signals (0.0 to 1.0)
    sparsity_score: float
    decay_score: float
    contradiction_score: float
    admission_score: float
    gap_score: float
    
    # Traceable evidence & suggested directions
    evidence_quotes: List[EvidenceQuote]
    suggested_directions: str
    
    # 2D Map Visualization Coordinates
    pca_x: float
    pca_y: float


class GapAnalysisRunResponse(BaseModel):
    run_id: str
    created_at: datetime
    total_papers: int
    num_clusters: int
    weights: Dict[str, float]
    ranked_gaps: List[ClusterGapResponse]


# ==========================================
# Expert Validation & Evaluation Schemas
# ==========================================
class ExpertFeedbackRequest(BaseModel):
    cluster_record_id: int
    expert_name: str = "Domain Expert"
    rating: str = Field(..., description="One of: 'genuine', 'already_studied', 'unclear'")
    notes: Optional[str] = ""


class ExpertFeedbackResponse(BaseModel):
    message: str
    validation_id: int
    current_precision: float
    total_rated: int


class BenchmarkQuestion(BaseModel):
    question: str
    expected_paper_id: str
    expected_answer_keywords: List[str]


class BenchmarkDetail(BaseModel):
    id: str
    category: str
    question: str
    intent: str
    relevant_chunks_in_corpus: int
    retrieved_relevant_top5: int
    retrieved_relevant_top10: int
    question_covered_at_5: bool = False
    distinct_papers_retrieved: int = 0
    multi_paper_supported: bool = False
    status: str = "evaluated"  # "evaluated" or "insufficient_corpus_evidence"
    explanation: Optional[str] = None


class EvaluationMetricsResponse(BaseModel):
    total_papers: int
    total_chunks: int
    question_coverage_at_5: Optional[float] = None
    question_coverage_at_10: Optional[float] = None
    multi_paper_evidence_rate: Optional[float] = None
    answer_faithfulness_rate: Optional[float] = None
    faithfulness_explanation: Optional[str] = None
    expert_gap_precision: Optional[float] = None
    strict_expert_precision: Optional[float] = None
    total_expert_validations: int
    validation_breakdown: Dict[str, int]
    benchmark_questions: Optional[List[Dict[str, Any]]] = None
    benchmark_details: Optional[List[BenchmarkDetail]] = None
