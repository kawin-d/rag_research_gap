import gc
import json
import os
import re
import stat
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import PDF_DIR, DATA_DIR, BASE_DIR
from app.database.session import get_db
from app.models.db_models import Paper, Chunk
from app.models.schemas import PaperResponse, PaperListResponse, ArxivSearchRequest
from app.services.paper_collector import PaperCollector
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import vector_store

router = APIRouter(prefix="/papers", tags=["Papers & Data Layer"])


@router.post("/search-arxiv", response_model=List[PaperResponse])
def search_and_ingest_arxiv(req: ArxivSearchRequest, db: Session = Depends(get_db)):
    """Search arXiv by topic or query, download PDFs, extract sections, chunk, and index embeddings."""
    try:
        raw_papers = PaperCollector.search_arxiv(
            query=req.query,
            max_results=req.max_results,
            download_pdfs=req.download_pdfs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"arXiv search failed: {str(e)}")

    if not raw_papers:
        return []

    saved_papers = []
    all_chunks_to_embed = []
    chunk_meta_list = []

    for item in raw_papers:
        # Check if paper already exists
        existing = db.query(Paper).filter(Paper.paper_id == item["paper_id"]).first()
        if existing:
            saved_papers.append(existing)
            continue

        paper = Paper(
            paper_id=item["paper_id"],
            title=item["title"],
            authors=json.dumps(item["authors"]),
            year=item["year"],
            abstract=item["abstract"],
            citation_count=item["citation_count"],
            url=item["url"],
            pdf_path=item["pdf_path"],
            sections_json=json.dumps(item["sections"])
        )
        db.add(paper)
        db.flush()

        # Add chunks
        for c in item["chunks"]:
            chunk_rec = Chunk(
                paper_id=paper.id,
                paper_uid=paper.paper_id,
                chunk_index=c["chunk_index"],
                section_name=c["section_name"],
                text=c["text"],
                token_count=c["token_count"]
            )
            db.add(chunk_rec)
            db.flush()

            all_chunks_to_embed.append(c["text"])
            chunk_meta_list.append({
                "chunk_id": chunk_rec.id,
                "paper_id": paper.paper_id,
                "paper_title": paper.title,
                "year": paper.year,
                "section_name": chunk_rec.section_name,
                "text": chunk_rec.text
            })

        saved_papers.append(paper)

    db.commit()

    # Compute and index vectors in background / batch
    if all_chunks_to_embed:
        embeddings = EmbeddingService.embed_texts(all_chunks_to_embed)
        vector_store.add_chunks(embeddings, chunk_meta_list)

    # Format output
    result = []
    for p in saved_papers:
        result.append(PaperResponse(
            id=p.id,
            paper_id=p.paper_id,
            title=p.title,
            authors=json.loads(p.authors) if p.authors else [],
            year=p.year,
            abstract=p.abstract or "",
            citation_count=p.citation_count or 0,
            url=p.url or "",
            pdf_path=p.pdf_path or "",
            sections=json.loads(p.sections_json) if p.sections_json else {},
            total_chunks=len(p.chunks) if p.chunks else 0,
            created_at=p.created_at
        ))

    return result


@router.post("/upload", response_model=PaperResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    year: Optional[int] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload a research paper PDF, extract sections, generate chunks, and update the vector index."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    try:
        parsed = PaperCollector.process_uploaded_pdf(contents, file.filename, title=title, year=year)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing error: {str(e)}")

    paper = Paper(
        paper_id=parsed["paper_id"],
        title=parsed["title"],
        authors=json.dumps(parsed["authors"]),
        year=parsed["year"],
        abstract=parsed["abstract"],
        citation_count=0,
        url="",
        pdf_path=parsed["pdf_path"],
        sections_json=json.dumps(parsed["sections"])
    )
    db.add(paper)
    db.flush()

    all_chunks_to_embed = []
    chunk_meta_list = []

    for c in parsed["chunks"]:
        chunk_rec = Chunk(
            paper_id=paper.id,
            paper_uid=paper.paper_id,
            chunk_index=c["chunk_index"],
            section_name=c["section_name"],
            text=c["text"],
            token_count=c["token_count"]
        )
        db.add(chunk_rec)
        db.flush()

        all_chunks_to_embed.append(c["text"])
        chunk_meta_list.append({
            "chunk_id": chunk_rec.id,
            "paper_id": paper.paper_id,
            "paper_title": paper.title,
            "year": paper.year,
            "section_name": chunk_rec.section_name,
            "text": chunk_rec.text
        })

    db.commit()

    if all_chunks_to_embed:
        embeddings = EmbeddingService.embed_texts(all_chunks_to_embed)
        vector_store.add_chunks(embeddings, chunk_meta_list)

    return PaperResponse(
        id=paper.id,
        paper_id=paper.paper_id,
        title=paper.title,
        authors=parsed["authors"],
        year=paper.year,
        abstract=paper.abstract or "",
        citation_count=0,
        url="",
        pdf_path=paper.pdf_path,
        sections=parsed["sections"],
        total_chunks=len(parsed["chunks"]),
        created_at=paper.created_at
    )


@router.get("", response_model=PaperListResponse)
@router.get("/", response_model=PaperListResponse)
def list_papers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List all research papers currently stored in the system."""
    total = db.query(Paper).count()
    papers = db.query(Paper).offset(skip).limit(limit).all()

    formatted = []
    for p in papers:
        formatted.append(PaperResponse(
            id=p.id,
            paper_id=p.paper_id,
            title=p.title,
            authors=json.loads(p.authors) if p.authors else [],
            year=p.year,
            abstract=p.abstract or "",
            citation_count=p.citation_count or 0,
            url=p.url or "",
            pdf_path=p.pdf_path or "",
            sections=json.loads(p.sections_json) if p.sections_json else {},
            total_chunks=len(p.chunks) if p.chunks else 0,
            created_at=p.created_at
        ))

    return PaperListResponse(total=total, papers=formatted)


def _safe_remove_file(file_path: Path) -> bool:
    """Robustly delete a file on Windows or Linux, handling permissions and read-only attributes."""
    try:
        if not file_path.exists() or not file_path.is_file():
            return False
        # Remove read-only attributes if present (common on Windows)
        try:
            os.chmod(str(file_path), stat.S_IWRITE | stat.S_IREAD)
        except Exception:
            pass
        file_path.unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f"[Papers] Warning: Failed to unlink {file_path}: {e}")
        return False


def _find_candidate_pdf_paths(path_str: str) -> List[Path]:
    """Find all possible filesystem locations for a stored pdf_path string."""
    if not path_str:
        return []
    candidates = []
    p = Path(path_str)
    candidates.append(p)
    if not p.is_absolute():
        candidates.append((BASE_DIR / p).resolve())
        candidates.append((DATA_DIR / p).resolve())
        candidates.append((PDF_DIR / p).resolve())
        candidates.append((PDF_DIR / p.name).resolve())
        candidates.append((DATA_DIR / p.name).resolve())
    else:
        candidates.append((PDF_DIR / p.name).resolve())
        candidates.append((DATA_DIR / p.name).resolve())
    return candidates


# Specific static paths placed BEFORE /{paper_id} to avoid 405 Method Not Allowed
@router.delete("", summary="Delete all papers and backend PDFs")
@router.delete("/", summary="Delete all papers and backend PDFs")
@router.post("", summary="Delete all papers and backend PDFs (POST)")
@router.post("/", summary="Delete all papers and backend PDFs (POST)")
@router.delete("/all", summary="Delete all papers and backend PDFs")
@router.post("/all", summary="Delete all papers and backend PDFs")
@router.delete("/delete-all", summary="Delete all papers and backend PDFs")
@router.post("/delete-all", summary="Delete all papers and backend PDFs")
@router.delete("/clear", summary="Delete all papers and backend PDFs")
@router.post("/clear", summary="Delete all papers and backend PDFs")
def delete_all_papers(db: Session = Depends(get_db)):
    """Remove all papers, chunks, vector embeddings, and all backend PDF files."""
    gc.collect()

    count = db.query(Paper).count()
    deleted_files_set = set()

    # 1. Clean up physical PDF files tracked in DB by papers
    papers = db.query(Paper).all()
    for p in papers:
        if p.pdf_path:
            for candidate in _find_candidate_pdf_paths(p.pdf_path):
                try:
                    resolved = candidate.resolve()
                    if resolved not in deleted_files_set and resolved.exists() and resolved.is_file():
                        if _safe_remove_file(resolved):
                            deleted_files_set.add(resolved)
                except Exception:
                    pass

    # 2. Clean up ALL remaining files and subdirectories in PDF_DIR
    if PDF_DIR.exists():
        for item in list(PDF_DIR.rglob("*")):
            if item.is_file():
                try:
                    resolved = item.resolve()
                    if resolved not in deleted_files_set:
                        if _safe_remove_file(resolved):
                            deleted_files_set.add(resolved)
                except Exception:
                    pass

        # Clean up any empty directories inside PDF_DIR
        for sub_dir in sorted(PDF_DIR.glob("**/*"), key=lambda p: len(str(p)), reverse=True):
            if sub_dir.is_dir() and sub_dir != PDF_DIR:
                try:
                    sub_dir.rmdir()
                except Exception:
                    pass

    # 3. Clean up any PDF files placed directly inside DATA_DIR
    if DATA_DIR.exists():
        for item in list(DATA_DIR.glob("*.pdf")) + list(DATA_DIR.glob("*.PDF")):
            if item.is_file():
                try:
                    resolved = item.resolve()
                    if resolved not in deleted_files_set:
                        if _safe_remove_file(resolved):
                            deleted_files_set.add(resolved)
                except Exception:
                    pass

    # 4. Remove all chunks and papers from database
    db.query(Chunk).delete()
    db.query(Paper).delete()
    db.commit()

    # 5. Reset vector store index and disk cache
    vector_store.clear()

    deleted_files = len(deleted_files_set)

    return {
        "message": f"Successfully deleted all {count} papers, {deleted_files} backend PDF files, and vector embeddings.",
        "deleted_count": count,
        "deleted_files_count": deleted_files
    }


@router.get("/{paper_id}", response_model=PaperResponse)
def get_paper(paper_id: str, db: Session = Depends(get_db)):
    """Get full metadata, sections, and chunk details for a specific paper."""
    paper = db.query(Paper).filter((Paper.paper_id == paper_id) | (Paper.id == paper_id)).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    return PaperResponse(
        id=paper.id,
        paper_id=paper.paper_id,
        title=paper.title,
        authors=json.loads(paper.authors) if paper.authors else [],
        year=paper.year,
        abstract=paper.abstract or "",
        citation_count=paper.citation_count or 0,
        url=paper.url or "",
        pdf_path=paper.pdf_path or "",
        sections=json.loads(paper.sections_json) if paper.sections_json else {},
        total_chunks=len(paper.chunks) if paper.chunks else 0,
        created_at=paper.created_at
    )


@router.get("/{paper_id}/pdf", summary="Download or view the local PDF file for a paper")
@router.get("/{paper_id}/download", summary="Download the local PDF file for a paper")
def get_paper_pdf(paper_id: str, db: Session = Depends(get_db)):
    """Serve the local PDF file associated with this paper if available."""
    paper = db.query(Paper).filter((Paper.paper_id == paper_id) | (Paper.id == paper_id)).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    if paper.pdf_path:
        for candidate in _find_candidate_pdf_paths(paper.pdf_path):
            try:
                resolved = candidate.resolve()
                if resolved.exists() and resolved.is_file():
                    safe_title = re.sub(r'[^\w\-\.]', '_', paper.title)[:50] or "paper"
                    return FileResponse(
                        str(resolved),
                        media_type="application/pdf",
                        filename=f"{safe_title}.pdf"
                    )
            except Exception:
                pass

    raise HTTPException(status_code=404, detail="Local PDF file not found on server for this paper.")


@router.delete("/{paper_id}")
@router.post("/{paper_id}/delete")
def delete_paper(paper_id: str, db: Session = Depends(get_db)):
    """Remove a paper and its chunks from the database, disk, and vector store."""
    gc.collect()

    paper = db.query(Paper).filter((Paper.paper_id == paper_id) | (Paper.id == paper_id)).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    paper_uid = paper.paper_id
    paper_title = paper.title
    pdf_path = paper.pdf_path

    # Clean up physical PDF file if present
    if pdf_path:
        for candidate in _find_candidate_pdf_paths(pdf_path):
            try:
                resolved = candidate.resolve()
                if resolved.exists() and resolved.is_file():
                    _safe_remove_file(resolved)
            except Exception:
                pass

    # Remove associated chunks from DB
    db.query(Chunk).filter(Chunk.paper_id == paper.id).delete()
    db.delete(paper)
    db.commit()

    # Remove associated chunks from vector store
    vector_store.remove_by_paper_id(paper_uid)

    return {"message": f"Paper '{paper_title}' successfully removed."}

