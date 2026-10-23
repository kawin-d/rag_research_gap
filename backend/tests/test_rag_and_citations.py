import sys
import json
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.session import SessionLocal, Base, engine
from app.models.db_models import Paper, Chunk
from app.services.pdf_processor import PDFProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import vector_store
from app.services.rag_service import RAGService


def setup_sample_data():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    sample_file = Path(__file__).resolve().parent.parent / "sample_data" / "sample_papers.json"
    with open(sample_file, "r", encoding="utf-8") as f:
        papers_data = json.load(f)

    all_chunks_to_embed = []
    chunk_meta_list = []

    for item in papers_data:
        existing = db.query(Paper).filter(Paper.paper_id == item["paper_id"]).first()
        if existing:
            continue

        paper = Paper(
            paper_id=item["paper_id"],
            title=item["title"],
            authors=json.dumps(item["authors"]),
            year=item.get("year", 2023),
            abstract=item.get("abstract", ""),
            citation_count=item.get("citation_count", 0),
            url=item.get("url", ""),
            pdf_path="",
            sections_json=json.dumps(item.get("sections", {}))
        )
        db.add(paper)
        db.flush()

        chunk_idx = 0
        for sec_name, sec_text in item.get("sections", {}).items():
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

    db.commit()

    if all_chunks_to_embed:
        embeddings = EmbeddingService.embed_texts(all_chunks_to_embed)
        vector_store.add_chunks(embeddings, chunk_meta_list)

    db.close()
    print(f"[Setup] Loaded papers into DB and indexed {len(chunk_meta_list)} chunks.")


def test_rag_query_with_citations():
    db = SessionLocal()
    question = "How does contextual calibration mitigate bias in few-shot prompt predictions?"
    print(f"\n[Test RAG] Asking: '{question}'")

    response = RAGService.query(db=db, question=question, top_k=3)
    print(f"Synthesized Answer:\n{response.answer}\n")
    print(f"Citations count: {len(response.citations)}")
    for c in response.citations:
        print(f"  - Citation: [{c.paper_title}, Year: {c.year}, Section: {c.section_name}]")

    assert len(response.retrieved_chunks) > 0, "No chunks retrieved"
    assert len(response.citations) > 0, "No citations generated"
    assert response.grounded is True, "Answer was not grounded"
    print("[PASS] test_rag_query_with_citations PASSED")
    db.close()


def test_rag_metadata_filtering():
    db = SessionLocal()
    # Filter only papers from 2020 or earlier
    question = "What happens to cross-domain sentiment analysis?"
    response = RAGService.query(db=db, question=question, top_k=3, filter_year_max=2020)
    print(f"\n[Test Metadata Filter] Retrieved {len(response.retrieved_chunks)} chunks with year <= 2020")
    for r in response.retrieved_chunks:
        print(f"  - Retrieved: {r.paper_title} (Year: {r.year}, Section: {r.section_name})")
        assert r.year <= 2020, f"Chunk year {r.year} exceeded filter 2020"

    print("[PASS] test_rag_metadata_filtering PASSED")
    db.close()


if __name__ == "__main__":
    setup_sample_data()
    test_rag_query_with_citations()
    test_rag_metadata_filtering()
