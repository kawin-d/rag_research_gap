import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.pdf_processor import PDFProcessor


def test_section_splitting():
    sample_academic_text = """
    A Survey of Few-Shot Learning in NLP
    
    Abstract
    Few-shot learning allows pre-trained models to rapidly adapt with scarce supervision.
    
    1. Introduction
    Recent years have witnessed remarkable breakthroughs in foundation models.
    However, prompt engineering is brittle.
    
    2. Methodology
    We compare soft continuous prompt tuning against hard prompts across 10 benchmarks.
    
    3. Results
    Soft prompts improve accuracy by 14% on classification benchmarks.
    
    4. Limitations and Future Work
    A noticeable limitation of our framework is the reliance on extensive hyperparameter tuning.
    We leave dynamic multi-task calibration for future work.
    
    5. Conclusion
    We provided a comprehensive empirical assessment of few-shot tuning.
    """

    sections = PDFProcessor.split_into_sections(sample_academic_text)
    print("Parsed Sections:", list(sections.keys()))
    assert "abstract" in sections, "Abstract not detected"
    assert "introduction" in sections, "Introduction not detected"
    assert "methodology" in sections, "Methodology not detected"
    assert "results" in sections, "Results not detected"
    assert "limitations_future_work" in sections, "Limitations/Future Work not detected"
    assert "conclusion" in sections, "Conclusion not detected"

    print("[PASS] test_section_splitting PASSED")


def test_chunking():
    text = "Word " * 600
    chunks = PDFProcessor.chunk_section("methodology", text, chunk_size=200, chunk_overlap=50)
    print(f"Generated {len(chunks)} chunks for 600 words.")
    assert len(chunks) >= 3, "Chunking should produce overlapping windows"
    assert chunks[0]["section_name"] == "methodology"
    print("[PASS] test_chunking PASSED")


if __name__ == "__main__":
    test_section_splitting()
    test_chunking()
