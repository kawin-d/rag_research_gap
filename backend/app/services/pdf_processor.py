import re
from typing import Dict, List, Tuple
from pathlib import Path

try:
    import pymupdf
    fitz = pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


# Standardized scientific sections
SECTION_PATTERNS = {
    "abstract": re.compile(r"^(?:abstract)\b", re.IGNORECASE),
    "introduction": re.compile(r"^(?:\d+[\.\s]+)?(?:introduction|background|overview)\b", re.IGNORECASE),
    "methodology": re.compile(r"^(?:\d+[\.\s]+)?(?:methodology|methods|method|proposed (?:method|approach|framework|model)|architecture|system model)\b", re.IGNORECASE),
    "results": re.compile(r"^(?:\d+[\.\s]+)?(?:results|experimental results|experiments|evaluation|empirical analysis)\b", re.IGNORECASE),
    "conclusion": re.compile(r"^(?:\d+[\.\s]+)?(?:conclusion|conclusions|concluding remarks|summary)\b", re.IGNORECASE),
    "limitations_future_work": re.compile(r"^(?:\d+[\.\s]+)?(?:limitations?|future work|open challenges?|threats to validity|broader impact)\b", re.IGNORECASE),
}


class PDFProcessor:
    """Extracts text from academic PDFs, splits into structured sections, and generates overlapping chunks."""

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract all raw text from a PDF file using PyMuPDF or PyPDF."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        full_text = []

        if PYMUPDF_AVAILABLE:
            try:
                with pymupdf.open(str(path)) as doc:
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        text = page.get_text("text")
                        if text:
                            full_text.append(text)
                if full_text:
                    return "\n\n".join(full_text)
            except Exception as e:
                print(f"[PDFProcessor] PyMuPDF extraction warning: {e}. Falling back to PyPDF.")

        if PYPDF_AVAILABLE:
            try:
                with open(str(path), "rb") as fh:
                    reader = pypdf.PdfReader(fh)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            full_text.append(extracted)
                if full_text:
                    return "\n\n".join(full_text)
            except Exception as e:
                print(f"[PDFProcessor] PyPDF extraction warning: {e}")

        if full_text:
            return "\n\n".join(full_text)
        return ""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean common PDF extraction artifacts: hyphenated line breaks, odd spaces, page headers."""
        # Fix hyphenated words broken across lines: "infor- \n mation" -> "information"
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
        # Normalize multiple newlines and spaces
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def split_into_sections(cls, raw_text: str) -> Dict[str, str]:
        """Split academic text into labelled sections based on common academic headings."""
        cleaned = cls.clean_text(raw_text)
        lines = cleaned.split("\n")

        sections: Dict[str, List[str]] = {
            "abstract": [],
            "introduction": [],
            "methodology": [],
            "results": [],
            "limitations_future_work": [],
            "conclusion": [],
            "general": []
        }

        current_section = "general"

        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue

            # Check if this line is a section heading (short line, matches heading pattern)
            is_heading = False
            if len(trimmed) < 80:
                for section_name, pattern in SECTION_PATTERNS.items():
                    if pattern.search(trimmed):
                        current_section = section_name
                        is_heading = True
                        break

            if not is_heading:
                # Check for explicit inline limitations/future work cues if in conclusion or results
                if current_section in ["results", "conclusion", "general"]:
                    if re.search(r"\b(?:we leave .* for future work|a limitation of (?:our|this)|future research should|unexplored)\b", trimmed, re.IGNORECASE):
                        sections["limitations_future_work"].append(trimmed)

                sections[current_section].append(trimmed)

        # Convert line lists to consolidated section text
        consolidated = {}
        for sec, text_lines in sections.items():
            content = " ".join(text_lines).strip()
            if content:
                consolidated[sec] = content

        # If abstract wasn't cleanly caught, attempt regex capture from beginning
        if "abstract" not in consolidated or len(consolidated["abstract"]) < 40:
            abs_match = re.search(r"(?:abstract|summary)\s*[:\n\-](.+?)(?:introduction|1\.\s+introduction|\n\n\d)", cleaned, re.IGNORECASE | re.DOTALL)
            if abs_match:
                consolidated["abstract"] = abs_match.group(1).strip()

        return consolidated

    @classmethod
    def chunk_section(cls, section_name: str, text: str, chunk_size: int = 400, chunk_overlap: int = 60) -> List[Dict[str, any]]:
        """Split section text into overlapping chunks of roughly chunk_size words."""
        words = text.split()
        if not words:
            return []

        chunks = []
        start_idx = 0
        chunk_idx = 0

        while start_idx < len(words):
            end_idx = min(start_idx + chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "chunk_index": chunk_idx,
                "section_name": section_name,
                "text": chunk_text,
                "token_count": len(chunk_words)
            })

            chunk_idx += 1
            if end_idx >= len(words):
                break
            start_idx += (chunk_size - chunk_overlap)

        return chunks

    @classmethod
    def process_paper_text(cls, raw_text: str, chunk_size: int = 400, chunk_overlap: int = 60) -> Tuple[Dict[str, str], List[Dict[str, any]]]:
        """Takes full paper text, returns parsed sections dict and all generated chunks."""
        sections = cls.split_into_sections(raw_text)
        all_chunks = []

        total_idx = 0
        for sec_name, sec_text in sections.items():
            sec_chunks = cls.chunk_section(sec_name, sec_text, chunk_size, chunk_overlap)
            for c in sec_chunks:
                c["chunk_index"] = total_idx
                all_chunks.append(c)
                total_idx += 1

        return sections, all_chunks
