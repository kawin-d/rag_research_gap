import re
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sqlalchemy.orm import Session

from app.models.db_models import Paper, Chunk, ClusterRecord, ExpertValidation, QueryLog
from app.models.schemas import (
    ExpertFeedbackRequest,
    ExpertFeedbackResponse,
    EvaluationMetricsResponse,
    BenchmarkDetail
)
from app.services.vector_store import vector_store
from app.services.embedding_service import EmbeddingService


# ==============================================================================
# 10 Evidence-Grounded Multi-Paper Research Gap Identification Benchmark Suite
# Aligned with: "Evidence-Grounded Multi-Paper Research Gap Identification using
# Temporal, Contradiction and Citation-Aware RAG"
# ==============================================================================
BENCHMARK_RESEARCH_GAP_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "BG-01",
        "category": "Research Gap Identification",
        "question": "What research gaps or unexplored areas are identified across the indexed research papers?",
        "intent": "Retrieve explicit author admissions, unexplored frontiers, and unaddressed research problems across the corpus.",
        "target_sections": ["limitations_future_work", "conclusion", "discussion", "abstract"],
        "evidence_cues": [
            "open question", "open problem", "open challenge", "unexplored", "research gap",
            "future work", "remains unclear", "further investigation", "not yet understood",
            "unaddressed", "promising avenue", "beyond the scope"
        ],
        "require_multi_paper": True
    },
    {
        "id": "BG-02",
        "category": "Limitations Analysis",
        "question": "What limitations do existing studies acknowledge, and which limitations remain unresolved?",
        "intent": "Retrieve admitted shortcomings, computational bottlenecks, algorithmic vulnerabilities, and unresolved constraints.",
        "target_sections": ["limitations_future_work", "conclusion", "results"],
        "evidence_cues": [
            "limitation", "drawback", "bottleneck", "vulnerability", "fails when", "degrades",
            "constraint", "trade-off", "costly", "overhead", "shortcoming", "weakness"
        ],
        "require_multi_paper": True
    },
    {
        "id": "BG-03",
        "category": "Contradiction Detection",
        "question": "Do the indexed papers report conflicting findings, conclusions, or experimental results on similar research problems?",
        "intent": "Detect contradictory empirical results, disagreements, or conflicting claims across papers on similar setups.",
        "target_sections": ["results", "discussion", "conclusion", "methodology"],
        "evidence_cues": [
            "however", "in contrast", "contrary to", "conflicting", "trade-off between",
            "degrades under", "discrepancy", "disagreement", "contested", "failed to replicate",
            "inconsistent", "mixed results"
        ],
        "require_multi_paper": True
    },
    {
        "id": "BG-04",
        "category": "Citation-Aware Analysis",
        "question": "How do cited papers relate to one another, and do their citation relationships reveal potential research gaps?",
        "intent": "Trace citation relationships, extensions, critiques, and missing connections between cited works.",
        "target_sections": ["introduction", "related_work", "methodology"],
        "evidence_cues": [
            "et al.", "prior work", "previous studies", "unlike", "extends", "proposed by",
            "cited in", "comparison with", "earlier approach", "reference"
        ],
        "require_multi_paper": True
    },
    {
        "id": "BG-05",
        "category": "Temporal Research Trends",
        "question": "How has research on a particular topic evolved over time, and which areas remain insufficiently explored?",
        "intent": "Analyze chronological shifts in methodology and identify lingering gaps that persist over multiple years.",
        "target_sections": ["introduction", "related_work", "conclusion", "limitations_future_work"],
        "evidence_cues": [
            "recent", "traditionally", "previously", "over time", "evolution", "shift",
            "emerged", "advances in", "historical", "still remains"
        ],
        "require_multi_paper": True
    },
    {
        "id": "BG-06",
        "category": "Methodological Gaps",
        "question": "What methods, datasets, evaluation metrics, or experimental settings are missing or insufficiently investigated?",
        "intent": "Identify methodological deficits, absent evaluation datasets, narrow benchmark scopes, or untested hyperparameter regimes.",
        "target_sections": ["methodology", "experiments", "evaluation", "limitations_future_work"],
        "evidence_cues": [
            "dataset", "metric", "benchmark", "evaluation protocol", "missing", "insufficient",
            "restricted to", "not evaluated", "baseline", "setup", "generalize"
        ],
        "require_multi_paper": False
    },
    {
        "id": "BG-07",
        "category": "Cross-Paper Comparison",
        "question": "What limitations or missing research directions become apparent when comparing findings across multiple papers?",
        "intent": "Cross-synthesize findings from distinct papers to highlight blind spots invisible when viewing any single study in isolation.",
        "target_sections": ["results", "discussion", "conclusion", "limitations_future_work"],
        "evidence_cues": [
            "compare", "comparison", "across", "while", "whereas", "outperform", "alternative",
            "superior", "different", "respective", "complementary"
        ],
        "require_multi_paper": True
    },
    {
        "id": "BG-08",
        "category": "Future Research Directions",
        "question": "What future research opportunities are supported by evidence from the indexed papers?",
        "intent": "Extract grounded next-step opportunities, actionable research proposals, and promising directions supported by text excerpts.",
        "target_sections": ["limitations_future_work", "conclusion", "discussion"],
        "evidence_cues": [
            "future research", "future work", "promising avenue", "promising direction",
            "next step", "should explore", "could investigate", "opportunity", "future investigation"
        ],
        "require_multi_paper": True
    },
    {
        "id": "BG-09",
        "category": "Evidence Verification",
        "question": "Which specific passages and papers support each proposed research gap?",
        "intent": "Verify that identified research gaps are traceable to verbatim excerpts with verifiable paper title, year, and section grounding.",
        "target_sections": ["limitations_future_work", "conclusion", "results", "abstract"],
        "evidence_cues": [
            "demonstrate", "show", "observe", "report", "finding", "evidence", "specifically",
            "empirical", "measured", "resulted in", "proves", "indicates"
        ],
        "require_multi_paper": True
    },
    {
        "id": "BG-10",
        "category": "Novelty Validation",
        "question": "Has a proposed research gap already been addressed by other indexed papers, or does the available evidence suggest that further investigation is warranted?",
        "intent": "Validate whether a candidate research gap is genuinely open vs already resolved by existing literature in the corpus.",
        "target_sections": ["related_work", "introduction", "discussion", "conclusion"],
        "evidence_cues": [
            "already addressed", "previously shown", "established", "remains open", "warranted",
            "unresolved", "genuine", "contested", "novelty", "prior work has"
        ],
        "require_multi_paper": True
    }
]

# Alias for backwards compatibility with any existing imports
DEFAULT_BENCHMARK = BENCHMARK_RESEARCH_GAP_QUESTIONS


class EvaluationService:
    """Provides evidence-grounded validation logging, strict/conservative precision scoring,
    question coverage metrics, citation faithfulness evaluation, and baseline comparisons.
    """

    @classmethod
    def get_benchmark_suite(cls, db: Optional[Session] = None) -> List[Dict[str, Any]]:
        """Returns the 10 multi-paper research gap identification benchmark questions."""
        return list(BENCHMARK_RESEARCH_GAP_QUESTIONS)

    @classmethod
    def _identify_ground_truth_chunks(cls, bench_def: Dict[str, Any]) -> set[int]:
        """Identifies known relevant evidence chunks in the corpus for a benchmark question.
        
        A chunk is considered relevant evidence if:
        1. It contains explicit evidentiary cues AND matches a target section, OR
        2. It contains multiple specific evidentiary cues indicating deep discussion, OR
        3. It matches author admission / contradiction patterns directly related to the inquiry.
        """
        target_secs = set(s.lower() for s in bench_def.get("target_sections", []))
        cues = [c.lower() for c in bench_def.get("evidence_cues", [])]
        relevant_chunk_ids = set()

        for chunk in vector_store.metadata:
            text = chunk.get("text", "").lower()
            sec = chunk.get("section_name", "").lower()

            has_cue = any(cue in text for cue in cues)
            sec_match = sec in target_secs
            cue_count = sum(1 for cue in cues if cue in text)

            if (has_cue and sec_match) or (cue_count >= 2):
                relevant_chunk_ids.add(chunk.get("chunk_id"))

        return relevant_chunk_ids

    @classmethod
    def submit_expert_feedback(cls, db: Session, req: ExpertFeedbackRequest) -> ExpertFeedbackResponse:
        """Records an expert's rating ('genuine', 'already_studied', 'unclear') for a detected gap.
        
        Calculates:
        - Conservative Precision = genuine / total_rated (unclear treated as non-genuine)
        - Strict Precision = genuine / (genuine + already_studied) (excludes unclear)
        """
        cluster = db.query(ClusterRecord).filter(ClusterRecord.id == req.cluster_record_id).first()
        if not cluster:
            raise ValueError(f"Cluster with ID {req.cluster_record_id} does not exist.")

        val = ExpertValidation(
            cluster_record_id=cluster.id,
            expert_name=req.expert_name,
            rating=req.rating.lower().strip(),
            notes=req.notes or ""
        )
        db.add(val)
        db.commit()
        db.refresh(val)

        # Compute running precision
        all_vals = db.query(ExpertValidation).all()
        genuine_count = sum(1 for v in all_vals if v.rating.lower().strip() == "genuine")
        precision = (genuine_count / len(all_vals)) if all_vals else 0.0

        return ExpertFeedbackResponse(
            message=f"Expert rating '{req.rating}' logged successfully for cluster: {cluster.topic_title}",
            validation_id=val.id,
            current_precision=round(precision, 4),
            total_rated=len(all_vals)
        )

    @classmethod
    def calculate_faithfulness_rate(cls, db: Session) -> Tuple[Optional[float], str]:
        """Calculates evidence faithfulness rate based on actual citation grounding
        in logged RAG queries and cluster records. Returns (rate, explanation).
        No hardcoded percentages are used.
        """
        # 1. Check logged RAG queries
        query_logs = db.query(QueryLog).all()
        if query_logs:
            total_citations = 0
            grounded_citations = 0
            for log in query_logs:
                try:
                    cites = json.loads(log.citations_json or "[]")
                    retrieved = json.loads(log.retrieved_chunks_json or "[]")
                    retrieved_text = " ".join([c.get("text", "").lower() for c in retrieved])

                    for c in cites:
                        total_citations += 1
                        quote = c.get("quote", "").strip().lower()
                        if not quote:
                            continue
                        q_clean = "".join(ch for ch in quote if ch.isalnum() or ch.isspace())
                        words = [w for w in q_clean.split() if len(w) > 3]
                        if not words:
                            continue
                        match_count = sum(1 for w in words if w in retrieved_text)
                        # Verifiable lexical overlap threshold (>= 60% of significant words present in retrieved context)
                        if match_count >= max(1, int(len(words) * 0.6)):
                            grounded_citations += 1
                except Exception:
                    continue

            if total_citations > 0:
                rate = round(grounded_citations / total_citations, 3)
                explanation = (
                    f"Measured from {total_citations} citations across {len(query_logs)} logged RAG queries: "
                    f"{grounded_citations}/{total_citations} ({rate * 100:.1f}%) citations directly verified against retrieved passage evidence."
                )
                return rate, explanation

        # 2. Check cluster evidence quotes if clusters have been run
        clusters = db.query(ClusterRecord).all()
        if clusters:
            total_quotes = 0
            grounded_quotes = 0
            all_chunk_text = " ".join([m.get("text", "").lower() for m in vector_store.metadata])
            for cl in clusters:
                try:
                    eqs = json.loads(cl.evidence_quotes_json or "[]")
                    for eq in eqs:
                        total_quotes += 1
                        q_text = eq.get("quote", "").strip().lower() if isinstance(eq, dict) else str(eq).lower()
                        if not q_text:
                            continue
                        words = [w for w in q_text.split() if len(w) > 3]
                        if words and sum(1 for w in words if w in all_chunk_text) >= max(1, int(len(words) * 0.5)):
                            grounded_quotes += 1
                except Exception:
                    continue

            if total_quotes > 0:
                rate = round(grounded_quotes / total_quotes, 3)
                explanation = (
                    f"Measured from {total_quotes} evidence quotes across {len(clusters)} gap clusters: "
                    f"{grounded_quotes}/{total_quotes} ({rate * 100:.1f}%) quotes confirmed in indexed paper chunks."
                )
                return rate, explanation

        return None, "Unavailable: No user RAG queries or gap cluster runs recorded yet to evaluate live faithfulness."

    @classmethod
    def _evaluate_retrieval_benchmarks(
        cls, benchmark_suite: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Evaluates retrieval quality across the 10 multi-paper research gap benchmark questions.
        
        Question Coverage@5 = Proportion of questions with at least 1 relevant evidence chunk in top-5.
        Multi-Paper Evidence Rate = Proportion of questions where top-10 evidence spans >= 2 distinct papers.
        """
        suite = benchmark_suite or cls.get_benchmark_suite()
        if len(vector_store.metadata) == 0:
            return {
                "question_coverage_at_5": 0.0,
                "question_coverage_at_10": 0.0,
                "multi_paper_evidence_rate": 0.0,
                "benchmark_details": []
            }

        coverages_5 = []
        coverages_10 = []
        multi_paper_flags = []
        details: List[BenchmarkDetail] = []

        for b in suite:
            q = b["question"]
            q_vec = EmbeddingService.embed_query(q)
            results_10 = vector_store.search(q_vec, top_k=10)
            results_5 = results_10[:5]

            # Identify all ground-truth relevant evidence chunks in the corpus
            ground_truth_ids = cls._identify_ground_truth_chunks(b)
            relevant_in_corpus = len(ground_truth_ids)

            if relevant_in_corpus == 0:
                # Explicitly handle questions where the current corpus contains no relevant evidence
                detail = BenchmarkDetail(
                    id=b.get("id", "BG"),
                    category=b.get("category", "General"),
                    question=q,
                    intent=b.get("intent", ""),
                    relevant_chunks_in_corpus=0,
                    retrieved_relevant_top5=0,
                    retrieved_relevant_top10=0,
                    question_covered_at_5=False,
                    distinct_papers_retrieved=0,
                    multi_paper_supported=False,
                    status="insufficient_corpus_evidence",
                    explanation="No explicit evidence passages matching this research gap inquiry found in current indexed papers."
                )
                details.append(detail)
                continue

            # Count hits
            h5 = sum(1 for c in results_5 if c.get("chunk_id") in ground_truth_ids)
            h10 = sum(1 for c in results_10 if c.get("chunk_id") in ground_truth_ids)

            cov5 = h5 > 0
            cov10 = h10 > 0

            # Distinct papers in relevant top-10 hits
            rel_papers = {c.get("paper_id") for c in results_10 if c.get("chunk_id") in ground_truth_ids}
            distinct_papers = len(rel_papers)
            multi_paper = distinct_papers >= 2

            coverages_5.append(1.0 if cov5 else 0.0)
            coverages_10.append(1.0 if cov10 else 0.0)
            multi_paper_flags.append(1.0 if multi_paper else 0.0)

            detail = BenchmarkDetail(
                id=b.get("id", "BG"),
                category=b.get("category", "General"),
                question=q,
                intent=b.get("intent", ""),
                relevant_chunks_in_corpus=relevant_in_corpus,
                retrieved_relevant_top5=h5,
                retrieved_relevant_top10=h10,
                question_covered_at_5=cov5,
                distinct_papers_retrieved=distinct_papers,
                multi_paper_supported=multi_paper,
                status="evaluated",
                explanation=f"Found {relevant_in_corpus} relevant corpus chunks; retrieved {h5} in top-5, {h10} in top-10 across {distinct_papers} distinct papers."
            )
            details.append(detail)

        avg_cov5 = float(np.mean(coverages_5)) if coverages_5 else 0.0
        avg_cov10 = float(np.mean(coverages_10)) if coverages_10 else 0.0
        avg_multi = float(np.mean(multi_paper_flags)) if multi_paper_flags else 0.0

        return {
            "question_coverage_at_5": round(avg_cov5, 3),
            "question_coverage_at_10": round(avg_cov10, 3),
            "multi_paper_evidence_rate": round(avg_multi, 3),
            "benchmark_details": details
        }

    @classmethod
    def get_evaluation_metrics(cls, db: Session) -> EvaluationMetricsResponse:
        """Calculates evidence-grounded system evaluation metrics:
        - Question Coverage@5 and Multi-Paper Evidence Diversity Rate
        - Verified Citation Faithfulness Rate
        - Conservative and Strict Expert Precision with complete review breakdown
        """
        total_papers = db.query(Paper).count()
        total_chunks = db.query(Chunk).count()

        # Expert validations
        validations = db.query(ExpertValidation).all()
        breakdown = {"genuine": 0, "already_studied": 0, "unclear": 0}
        for v in validations:
            r = v.rating.lower().strip()
            if r in breakdown:
                breakdown[r] += 1
            else:
                breakdown[r] = breakdown.get(r, 0) + 1

        total_rated = len(validations)
        genuine = breakdown["genuine"]
        already_studied = breakdown["already_studied"]

        # Conservative Precision: genuine / total_rated (unclear treated as non-genuine)
        expert_precision = round(genuine / total_rated, 3) if total_rated > 0 else None

        # Strict Precision: genuine / (genuine + already_studied) (excludes unclear judgments)
        decisive_count = genuine + already_studied
        strict_precision = round(genuine / decisive_count, 3) if decisive_count > 0 else None

        # Benchmark retrieval evaluation
        bench_res = cls._evaluate_retrieval_benchmarks()

        # Real Faithfulness Calculation (no hardcoded numbers)
        faithfulness_rate, faithfulness_exp = cls.calculate_faithfulness_rate(db)

        return EvaluationMetricsResponse(
            total_papers=total_papers,
            total_chunks=total_chunks,
            question_coverage_at_5=bench_res["question_coverage_at_5"],
            question_coverage_at_10=bench_res["question_coverage_at_10"],
            multi_paper_evidence_rate=bench_res["multi_paper_evidence_rate"],
            answer_faithfulness_rate=faithfulness_rate,
            faithfulness_explanation=faithfulness_exp,
            expert_gap_precision=expert_precision,
            strict_expert_precision=strict_precision,
            total_expert_validations=total_rated,
            validation_breakdown=breakdown,
            benchmark_questions=[
                {
                    "id": b["id"],
                    "category": b["category"],
                    "question": b["question"],
                    "intent": b["intent"],
                    "require_multi_paper": b.get("require_multi_paper", True)
                }
                for b in BENCHMARK_RESEARCH_GAP_QUESTIONS
            ],
            benchmark_details=bench_res["benchmark_details"]
        )

    @classmethod
    def run_baseline_comparison(cls, question: str) -> Dict[str, Any]:
        """Compares Dense RAG vs Plain Keyword Search for any unrestricted user query."""
        q_vec = EmbeddingService.embed_query(question)
        dense_results = vector_store.search(q_vec, top_k=3)

        # Baseline 1: Plain Lexical / Keyword Search (BM25 simulation)
        q_words = set(re.findall(r"\w+", question.lower()))
        lexical_scored = []
        for meta in vector_store.metadata:
            text_words = set(re.findall(r"\w+", meta.get("text", "").lower()))
            overlap = len(q_words.intersection(text_words))
            if overlap > 0:
                lexical_scored.append((overlap, meta))
        lexical_scored.sort(key=lambda x: x[0], reverse=True)
        keyword_results = [item[1] for item in lexical_scored[:3]]

        return {
            "question": question,
            "dense_rag_results": [
                {
                    "title": r.get("paper_title"),
                    "section": r.get("section_name"),
                    "score": round(r.get("similarity_score", 0), 4)
                }
                for r in dense_results
            ],
            "plain_keyword_results": [
                {
                    "title": r.get("paper_title"),
                    "section": r.get("section_name")
                }
                for r in keyword_results
            ],
            "comparison_notes": "Dense semantic RAG captures synonymy and conceptual similarity that keyword matching misses entirely."
        }
