import os
import re
import time
import uuid
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional
import arxiv
from app.config import PDF_DIR, settings
from app.services.pdf_processor import PDFProcessor


class PaperCollector:
    """Collects research papers and metadata from arXiv, Semantic Scholar, or local PDF uploads."""

    @classmethod
    def search_arxiv(cls, query: str, max_results: int = 10, download_pdfs: bool = True) -> List[Dict[str, Any]]:
        """Search arXiv and download PDFs, extracting metadata and parsed sections."""
        clean_query = (query or "").strip()
        if not clean_query:
            return []

        # Detect if the query is a specific arXiv ID or arXiv URL
        arxiv_id_match = re.search(
            r'(?:arxiv\.org/(?:abs|pdf)/|arxiv:)?([a-zA-Z\-]+(?:\.[a-zA-Z\-]+)?/\d{7}|\d{4}\.\d{4,5}(?:v\d+)?)',
            clean_query,
            re.IGNORECASE
        )

        client = arxiv.Client(
            page_size=max_results,
            delay_seconds=3.0,
            num_retries=3
        )

        if arxiv_id_match and (clean_query.lower().startswith("http") or clean_query.lower().startswith("arxiv:") or re.fullmatch(r"[\d\.\w\-\/]+", clean_query)):
            clean_target_id = arxiv_id_match.group(1).rstrip(".pdf")
            search = arxiv.Search(id_list=[clean_target_id])
        else:
            search = arxiv.Search(
                query=clean_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

        PDF_DIR.mkdir(parents=True, exist_ok=True)

        arxiv_results = []
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                arxiv_results = list(client.results(search))
                break
            except Exception as e:
                print(f"[PaperCollector] arXiv query attempt {attempt + 1}/{max_attempts} failed: {e}")
                if attempt == max_attempts - 1:
                    raise
                time.sleep(2.0 * (attempt + 1))

        results = []
        for result in arxiv_results:
            paper_id = result.get_short_id()
            clean_id = re.sub(r"[^\w\-\.]", "_", paper_id)
            pdf_filename = f"arxiv_{clean_id}.pdf"
            pdf_path = PDF_DIR / pdf_filename

            # Download PDF if requested and not yet cached
            if download_pdfs:
                try:
                    if not pdf_path.exists():
                        download_candidates = []
                        if result.pdf_url:
                            download_candidates.append(result.pdf_url)
                        clean_no_v = re.sub(r"v\d+$", "", clean_id)
                        fallback_url = f"https://arxiv.org/pdf/{clean_no_v}.pdf"
                        if fallback_url not in download_candidates:
                            download_candidates.append(fallback_url)

                        headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                            "Accept": "application/pdf,application/octet-stream,*/*"
                        }
                        with httpx.Client(timeout=45.0, follow_redirects=True) as h_client:
                            for dl_url in download_candidates:
                                try:
                                    resp = h_client.get(dl_url, headers=headers)
                                    if resp.status_code == 200 and len(resp.content) > 1000:
                                        with open(pdf_path, "wb") as f_out:
                                            f_out.write(resp.content)
                                        print(f"[PaperCollector] Successfully downloaded PDF: {pdf_filename} ({len(resp.content)} bytes)")
                                        break
                                    else:
                                        print(f"[PaperCollector] PDF download returned status {resp.status_code} for {dl_url}")
                                except Exception as dl_err:
                                    print(f"[PaperCollector] Error downloading from {dl_url}: {dl_err}")
                except Exception as e:
                    print(f"[PaperCollector] PDF download failed for {paper_id}: {e}")

            # Get publication year
            year = result.published.year if result.published else None

            # Authors list
            authors = [a.name for a in result.authors]

            # Try to fetch additional citation count from Semantic Scholar if available
            citation_count = cls.fetch_semantic_scholar_citations(result.title)

            # Process text if PDF exists
            sections = {"abstract": result.summary}
            chunks = []
            if pdf_path.exists():
                try:
                    raw_text = PDFProcessor.extract_text_from_pdf(str(pdf_path))
                    parsed_sections, parsed_chunks = PDFProcessor.process_paper_text(
                        raw_text, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
                    )
                    if parsed_sections:
                        sections.update(parsed_sections)
                    chunks = parsed_chunks
                except Exception as e:
                    print(f"[PaperCollector] Error parsing downloaded PDF {pdf_path}: {e}")

            # If no PDF was downloaded or parsed, create at least an abstract chunk
            if not chunks and result.summary:
                chunks.append({
                    "chunk_index": 0,
                    "section_name": "abstract",
                    "text": result.summary,
                    "token_count": len(result.summary.split())
                })

            results.append({
                "paper_id": f"arxiv_{clean_id}",
                "title": result.title.replace("\n", " ").strip(),
                "authors": authors,
                "year": year,
                "abstract": result.summary.replace("\n", " ").strip(),
                "citation_count": citation_count,
                "url": result.entry_id,
                "pdf_path": str(pdf_path.resolve()) if pdf_path.exists() else "",
                "sections": sections,
                "chunks": chunks
            })

        return results

    @classmethod
    def fetch_semantic_scholar_citations(cls, title: str) -> int:
        """Fetch citation count from Semantic Scholar Open API."""
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            headers = {}
            if settings.SEMANTIC_SCHOLAR_API_KEY:
                headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY

            params = {
                "query": title[:150],
                "limit": 1,
                "fields": "citationCount"
            }
            with httpx.Client(timeout=4.0) as client:
                res = client.get(url, params=params, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    if data.get("total", 0) > 0 and "data" in data and len(data["data"]) > 0:
                        return data["data"][0].get("citationCount", 0) or 0
        except Exception:
            pass
        return 0

    @classmethod
    def process_uploaded_pdf(cls, file_bytes: bytes, filename: str, title: Optional[str] = None, year: Optional[int] = None) -> Dict[str, Any]:
        """Save and parse an uploaded PDF file."""
        clean_name = re.sub(r"[^\w\-\.]", "_", filename)
        file_id = f"custom_{uuid.uuid4().hex[:8]}"
        target_path = PDF_DIR / f"{file_id}_{clean_name}"

        try:
            with open(target_path, "wb") as f:
                f.write(file_bytes)

            # Extract raw text
            raw_text = PDFProcessor.extract_text_from_pdf(str(target_path))
            sections, chunks = PDFProcessor.process_paper_text(
                raw_text, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
            )
        except Exception:
            if target_path.exists():
                try:
                    target_path.unlink()
                except Exception:
                    pass
            raise

        # Heuristic title detection if not provided
        inferred_title = title
        if not inferred_title:
            # First lines of paper often contain title
            lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
            inferred_title = lines[0] if lines else filename.replace(".pdf", "").replace("_", " ")

        abstract = sections.get("abstract", "")
        if not abstract and chunks:
            abstract = chunks[0]["text"][:600]

        return {
            "paper_id": file_id,
            "title": inferred_title,
            "authors": ["Uploaded Paper"],
            "year": year or 2024,
            "abstract": abstract,
            "citation_count": 0,
            "url": "",
            "pdf_path": str(target_path),
            "sections": sections,
            "chunks": chunks
        }
