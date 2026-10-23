from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database.session import engine, Base
from app.routers import papers, rag, gaps, evaluation, sample

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
## Final Year Project: RAG-Based Research Paper Analysis and Research Gap Identification System

### Core Features:
1. **Data Layer**: Search arXiv API, query Semantic Scholar, upload custom PDFs, and store metadata in SQLite.
2. **Processing Layer**: Extract academic sections (*Abstract, Introduction, Methodology, Results, Conclusion, Limitations/Future Work*) and generate overlapping chunks with source tags.
3. **Embedding & Vector Storage**: Dense semantic embeddings (*all-MiniLM-L6-v2*) with persistent vector index and multi-attribute metadata filtering.
4. **Research Paper Analysis (RAG Layer)**: Answer scientific questions strictly grounded in retrieved chunks, with explicit traceability citations `[Paper, Year, Section]`.
5. **Research Gap Identification Layer**: Topic clustering + computation of 4 independent gap signals:
   - *Semantic Sparsity*
   - *Temporal Decay*
   - *Contradiction Density*
   - *Explicit Author Admission*
   Combined into a weighted **Gap Score** with 2D PCA/UMAP coordinates and evidence quotes.
6. **Evaluation Layer**: Retrieval Recall@k, answer faithfulness check, baseline comparison against keyword search, and expert validation precision.
    """
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers under /api
app.include_router(papers.router, prefix=settings.API_V1_PREFIX)
app.include_router(rag.router, prefix=settings.API_V1_PREFIX)
app.include_router(gaps.router, prefix=settings.API_V1_PREFIX)
app.include_router(evaluation.router, prefix=settings.API_V1_PREFIX)
app.include_router(sample.router, prefix=settings.API_V1_PREFIX)

# Also mount routers directly so calls without /api prefix resolve properly
app.include_router(papers.router)
app.include_router(rag.router)
app.include_router(gaps.router)
app.include_router(evaluation.router)
app.include_router(sample.router)


from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/ui", tags=["Frontend UI"])
def serve_ui():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            }
        )
    return {"message": "UI not found"}

@app.get("/", tags=["Health & Status"])
def root():
    return {
        "status": "online",
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "ui_url": "/ui",
        "docs_url": "/docs",
        "endpoints": {
            "search_arxiv": "/api/papers/search-arxiv",
            "upload_pdf": "/api/papers/upload",
            "list_papers": "/api/papers",
            "rag_qa": "/api/rag/ask",
            "retrieve_chunks": "/api/rag/retrieve",
            "analyze_gaps": "/api/gaps/analyze",
            "latest_clusters": "/api/gaps/clusters",
            "gap_report": "/api/gaps/report/{cluster_id}",
            "expert_feedback": "/api/evaluation/expert-feedback",
            "metrics": "/api/evaluation/metrics",
            "load_sample_data": "/api/sample/load"
        }
    }

