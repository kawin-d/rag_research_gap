# RAG-Based Research Paper Analysis and Research Gap Identification System (Backend)

Final Year Project: High-Performance FastAPI Backend for Grounded Research Paper Question Answering and Multi-Signal Scientific Gap Discovery.

---

## 1. System Overview

This backend is a specialized academic research platform implementing **Retrieval-Augmented Generation (RAG)** and **Unsupervised Research Gap Identification**. It transforms collections of research papers into structured, searchable knowledge and surfaces genuinely under-explored scientific domains using traceable evidence.

### Two Core Capabilities:
1. **Research Paper Analysis (RAG):** Answers scientific inquiries strictly grounded in retrieved document excerpts, outputting verified citations specifying the paper title, year, section name, and exact supporting quote.
2. **Research Gap Identification:** Groups papers into thematic clusters via dense embeddings and evaluates each cluster across **four independent gap signals** (Semantic Sparsity, Temporal Decay, Contradiction Density, and Explicit Author Admission) to compute a traceable, ranked **Gap Score** complete with 2D map coordinates for visualization.

---

## 2. Architecture & Layer Design

```
                     ┌───────────────────────────────────────────────┐
                     │           Client / Frontend App               │
                     └──────────────────────┬────────────────────────┘
                                            │ REST API (JSON / Markdown)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                FastAPI Backend Application                             │
│                                                                                        │
│  [Routers]                                                                             │
│  ├── /api/papers      -> arXiv search, PDF upload, section viewer, metadata store     │
│  ├── /api/rag         -> Question answering, top-k retrieval, citations, chunk logs    │
│  ├── /api/gaps        -> Cluster gap analysis, 4 signals, 2D coordinates, reports      │
│  ├── /api/evaluation  -> Expert validation, precision tracking, baseline comparisons   │
│  └── /api/sample      -> One-click Few-Shot Learning scientific paper dataset loader   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        ▼                                   ▼                                   ▼
┌──────────────────┐               ┌──────────────────┐               ┌──────────────────┐
│  Services Layer  │               │   Vector Store   │               │   Database Layer │
│  - PaperCollector│               │   & Embeddings   │               │   (SQLite)       │
│  - PDFProcessor  │               │                  │               │                  │
│  - EmbeddingSvc  │──────────────▶│ - MiniLM-L6-v2   │               │ - papers         │
│  - VectorStore   │               │ - Cosine sim     │               │ - chunks         │
│  - LLMService    │               │ - Metadata filter│               │ - gap_runs       │
│  - GapAnalyzer   │               │ - Persistence    │               │ - cluster_records│
│  - EvalService   │               └──────────────────┘               │ - expert_valids  │
└──────────────────┘                                                  │ - query_logs     │
                                                                      └──────────────────┘
```

---

## 3. The 4 Research Gap Signals

Rather than relying on subjective LLM speculation, the Gap Score is calculated from **4 orthogonal mathematical & empirical signals**:

$$\text{Gap Score} = w_1 \cdot S_{\text{sparsity}} + w_2 \cdot S_{\text{decay}} + w_3 \cdot S_{\text{contradiction}} + w_4 \cdot S_{\text{admission}}$$

### 1. Semantic Sparsity ($S_{\text{sparsity}}$)
- **Concept:** Measures how isolated and small a topic cluster is relative to the general literature mass.
- **Formula:** Combination of normalized cluster size ratio and Euclidean distance from the cluster centroid to the global corpus centroid:
  $$S_{\text{sparsity}} = 0.6 \left(1 - \frac{N_{\text{cluster}}}{N_{\text{total}}}\right) + 0.4 \min\left(\|\mathbf{c}_{\text{cluster}} - \mathbf{c}_{\text{global}}\|, 1.0\right)$$

### 2. Temporal Decay ($S_{\text{decay}}$)
- **Concept:** Differentiates between dead/abandoned topics and active trends. A topic that commanded strong publication momentum in earlier years and abruptly stalled in recent years indicates an unresolved bottleneck.
- **Formula:** Ratio of historical publications ($t < t_{\text{recent}}$) versus active recent publications ($t \ge t_{\text{recent}}$).

### 3. Contradiction Density ($S_{\text{contradiction}}$)
- **Concept:** Unresolved debates, conflicting findings, or performance trade-offs highlight open problems.
- **Detection:** Scans text chunks for tension markers (*however, in contrast, conflicting results, degrades under, trade-off between, failed to replicate*) and computes frequency across papers.

### 4. Explicit Author Admission ($S_{\text{admission}}$)
- **Concept:** High-credibility admissions where researchers explicitly acknowledge unaddressed questions in their *Limitations* and *Future Work* sections.
- **Detection:** Parses chunks from sections `limitations_future_work` and `conclusion` for phrases like *"remains an open question"*, *"we leave for future work"*, *"a limitation of our method"*, and *"unexplored"*.

---

## 4. Directory Structure

```
backend/
├── app/
│   ├── main.py                  # FastAPI application entrypoint with CORS
│   ├── config.py                # Configuration and environment parameters
│   ├── database/
│   │   └── session.py           # SQLAlchemy SQLite connection and session factory
│   ├── models/
│   │   ├── db_models.py         # SQLAlchemy tables (Paper, Chunk, ClusterRecord, etc.)
│   │   └── schemas.py           # Pydantic v2 validation schemas
│   ├── services/
│   │   ├── paper_collector.py   # arXiv API, Semantic Scholar citations, PDF upload
│   │   ├── pdf_processor.py     # PyMuPDF section parser & overlapping chunk generator
│   │   ├── embedding_service.py # sentence-transformers (all-MiniLM-L6-v2) + cache
│   │   ├── vector_store.py      # High-speed vector index with metadata filtering
│   │   ├── llm_service.py       # Pluggable LLM (OpenAI, Anthropic, Gemini, Ollama, Heuristic)
│   │   ├── rag_service.py       # Grounded RAG retrieval, citation builder, query logger
│   │   ├── gap_analyzer.py      # KMeans clustering, 4 signals, 2D PCA projection
│   │   └── evaluation_service.py# Recall@k, faithfulness, baseline comparison, expert validation
│   └── routers/
│       ├── papers.py            # /api/papers endpoints
│       ├── rag.py               # /api/rag endpoints
│       ├── gaps.py              # /api/gaps endpoints
│       ├── evaluation.py        # /api/evaluation endpoints
│       └── sample.py            # /api/sample endpoints
├── data/
│   ├── pdfs/                    # Downloaded and uploaded PDF documents
│   ├── vector_cache/            # Persistent vector index (.npz) & metadata (.json)
│   └── research_system.db       # SQLite relational database
├── sample_data/
│   └── sample_papers.json       # Pre-structured Few-Shot Learning paper corpus
├── tests/
│   ├── test_pdf_processor.py    # Unit tests for section extraction and chunking
│   ├── test_rag_and_citations.py# Integration test for RAG, citations, and metadata filters
│   ├── test_gap_analyzer.py     # Integration test for clustering and the 4 gap signals
│   ├── test_evaluation.py       # Tests for expert feedback, precision, and recall
│   └── test_api_endpoints.py    # Complete FastAPI client REST test suite
├── .env.example                 # Environment variables template
├── requirements.txt             # Python dependencies
└── run.py                       # Server runner (uvicorn app.main:app)
```

---

## 5. Quickstart Guide

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
> *Note:* An API key is **not required** to test the system! The backend comes with a built-in offline scientific grounding engine that produces cited answers and detects contradictions out-of-the-box. If you add `OPENAI_API_KEY` or `GEMINI_API_KEY`, the system automatically switches to the cloud LLM.

### 3. Start the Server
```bash
python run.py
```
The server host and port can be customized in `.env` via `HOST` (default: `0.0.0.0`) and `PORT` (default: `8000`).
By default, the server starts at:
- **Interactive Web UI:** `http://127.0.0.1:8000/ui`
- **Interactive Swagger Documentation:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

---

## 6. API Reference & Usage Examples

### A. Load Pre-Packaged Scientific Dataset
Immediately loads 6 curated papers on Few-Shot Learning with full sections, temporal spread, and admissions:
```bash
curl -X POST "http://127.0.0.1:8000/api/sample/load"
```

### B. Search & Ingest from arXiv API
```bash
curl -X POST "http://127.0.0.1:8000/api/papers/search-arxiv" \
     -H "Content-Type: application/json" \
     -d '{"query": "few-shot prompt learning in NLP", "max_results": 5, "download_pdfs": true}'
```

### C. Ask a Research Question (RAG with Citations)
```bash
curl -X POST "http://127.0.0.1:8000/api/rag/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "How does contextual calibration mitigate bias in few-shot prompt predictions?",
       "top_k": 3,
       "filter_sections": ["abstract", "methodology", "results"]
     }'
```
**Sample Response:**
```json
{
  "question": "How does contextual calibration mitigate bias in few-shot prompt predictions?",
  "answer": "Based on the retrieved research papers: Contextual calibration boosts few-shot accuracy by up to 30% across multiple benchmarks without requiring any gradient updates or extra training parameters. [Paper: Calibrate Before Use: Mitigating Majority and Recency Bias in Few-Shot Learning, Year: 2021, Section: results] ...",
  "citations": [
    {
      "paper_id": "paper_fsl_03",
      "paper_title": "Calibrate Before Use: Mitigating Majority and Recency Bias in Few-Shot Learning",
      "year": 2021,
      "section_name": "results",
      "chunk_id": 16,
      "quote": "Contextual calibration boosts few-shot accuracy by up to 30%..."
    }
  ],
  "retrieved_chunks": [...],
  "grounded": true,
  "context_used": 3
}
```

### D. Run Research Gap Analysis
```bash
curl -X POST "http://127.0.0.1:8000/api/gaps/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "weights": {
         "w_sparsity": 0.25,
         "w_decay": 0.25,
         "w_contradiction": 0.25,
         "w_admission": 0.25
       },
       "num_clusters": 3
     }'
```
Returns ranked clusters with signal scores, 2D coordinates `(pca_x, pca_y)` for scatter plots, evidence quotes, and synthesized future research directions.

### E. Download Gap Report as Markdown
```bash
curl -X GET "http://127.0.0.1:8000/api/gaps/report/1?format=markdown"
```

### F. Submit Expert Validation Feedback
Domain expert rates a detected gap as `genuine`, `already_studied`, or `unclear`:
```bash
curl -X POST "http://127.0.0.1:8000/api/evaluation/expert-feedback" \
     -H "Content-Type: application/json" \
     -d '{
       "cluster_record_id": 1,
       "expert_name": "Prof. Turing",
       "rating": "genuine",
       "notes": "Verified open challenge in out-of-distribution adversarial robustness."
     }'
```

### G. View System Evaluation Metrics
```bash
curl -X GET "http://127.0.0.1:8000/api/evaluation/metrics"
```

---

## 7. Running the Test Suite

Run all verification tests individually:
```bash
python backend/tests/test_pdf_processor.py
python backend/tests/test_rag_and_citations.py
python backend/tests/test_gap_analyzer.py
python backend/tests/test_evaluation.py
python backend/tests/test_api_endpoints.py
```
All tests verify zero regressions across section parsing, vector search, citation grounding, clustering, signal computation, and API endpoints.
