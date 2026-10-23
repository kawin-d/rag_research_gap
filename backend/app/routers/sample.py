import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.db_models import Paper, Chunk
from app.services.pdf_processor import PDFProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import vector_store

router = APIRouter(prefix="/sample", tags=["Sample Data Layer"])

SAMPLE_FILE = Path(__file__).resolve().parent.parent.parent / "sample_data" / "sample_papers.json"


@router.post("/load")
def load_sample_papers(reset: bool = False, db: Session = Depends(get_db)):
    """Load curated academic papers on 'Few-Shot Learning in NLP' into SQLite and index their vectors.
    
    If reset=True, removes existing papers and vector index before loading the 6 benchmark papers.
    """
    if not SAMPLE_FILE.exists():
        raise HTTPException(status_code=404, detail="sample_papers.json not found")

    with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
        papers_data = json.load(f)

    if reset:
        # Wipe existing DB chunks and papers
        db.query(Chunk).delete()
        db.query(Paper).delete()
        db.commit()
        # Wipe vector store
        vector_store.clear()

    loaded_count = 0
    all_chunks_to_embed = []
    chunk_meta_list = []

    for item in papers_data:
        existing = db.query(Paper).filter(Paper.paper_id == item["paper_id"]).first()
        if existing:
            continue

        sections = item.get("sections", {})
        # Create Paper
        paper = Paper(
            paper_id=item["paper_id"],
            title=item["title"],
            authors=json.dumps(item["authors"]),
            year=item.get("year", 2023),
            abstract=item.get("abstract", ""),
            citation_count=item.get("citation_count", 0),
            url=item.get("url", ""),
            pdf_path="",
            sections_json=json.dumps(sections)
        )
        db.add(paper)
        db.flush()

        # Generate chunks for each section
        chunk_idx = 0
        for sec_name, sec_text in sections.items():
            sec_chunks = PDFProcessor.chunk_section(sec_name, sec_text, chunk_size=250, chunk_overlap=40)
            for sc in sec_chunks:
                chunk_rec = Chunk(
                    paper_id=paper.id,
                    paper_uid=paper.paper_id,
                    chunk_index=chunk_idx,
                    section_name=sec_name,
                    text=sc["text"],
                    token_count=sc["token_count"]
                )
                db.add(chunk_rec)
                db.flush()

                all_chunks_to_embed.append(sc["text"])
                chunk_meta_list.append({
                    "chunk_id": chunk_rec.id,
                    "paper_id": paper.paper_id,
                    "paper_title": paper.title,
                    "year": paper.year,
                    "section_name": sec_name,
                    "text": sc["text"]
                })
                chunk_idx += 1

        loaded_count += 1

    db.commit()

    if all_chunks_to_embed:
        embeddings = EmbeddingService.embed_texts(all_chunks_to_embed)
        vector_store.add_chunks(embeddings, chunk_meta_list)

    total_papers = db.query(Paper).count()
    total_chunks = db.query(Chunk).count()

    return {
        "message": f"Successfully loaded {loaded_count} research papers (reset={reset}).",
        "total_papers_in_db": total_papers,
        "total_chunks_in_vector_store": total_chunks,
        "topics_covered": "Few-Shot Prompt Learning, Prototypical Networks, LoRA, Contextual Calibration, Adversarial Robustness, Cross-Lingual Transfer"
    }


@router.post("/reset")
def reset_to_benchmark(db: Session = Depends(get_db)):
    """Reset the entire corpus and vector index back to the 6 benchmark Few-Shot NLP papers."""
    return load_sample_papers(reset=True, db=db)


from app.routers.papers import delete_all_papers


@router.post("/clear")
def clear_all_papers(db: Session = Depends(get_db)):
    """Completely wipe all papers, chunks, vector index, and all backend PDF files."""
    return delete_all_papers(db=db)
