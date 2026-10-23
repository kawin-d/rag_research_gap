import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.session import SessionLocal
from app.models.db_models import ClusterRecord
from app.models.schemas import ExpertFeedbackRequest
from app.services.evaluation_service import EvaluationService


def test_expert_validation_and_metrics():
    db = SessionLocal()
    print("\n--- Testing Expert Validation & Precision Metrics ---")

    # Pick first cluster
    first_cluster = db.query(ClusterRecord).first()
    assert first_cluster is not None, "No cluster record found"

    # Expert submits rating: 'genuine'
    req = ExpertFeedbackRequest(
        cluster_record_id=first_cluster.id,
        expert_name="Prof. Alan Turing",
        rating="genuine",
        notes="Strong evidence of adversarial vulnerability and lack of calibration in few-shot regimes."
    )

    res = EvaluationService.submit_expert_feedback(db=db, req=req)
    print(f"Feedback logged: {res.message}")
    print(f"Running Precision: {res.current_precision * 100:.1f}% across {res.total_rated} evaluations.")
    assert res.current_precision > 0.0

    # Retrieve system metrics
    metrics = EvaluationService.get_evaluation_metrics(db=db)
    print("\nSystem Evaluation Metrics:")
    print(f"  Total Papers: {metrics.total_papers}")
    print(f"  Total Chunks: {metrics.total_chunks}")
    print(f"  Question Coverage@5: {metrics.question_coverage_at_5}")
    print(f"  Multi-Paper Evidence Rate: {metrics.multi_paper_evidence_rate}")
    print(f"  Answer Faithfulness Rate: {metrics.answer_faithfulness_rate} ({metrics.faithfulness_explanation})")
    print(f"  Conservative Expert Precision: {metrics.expert_gap_precision}")
    print(f"  Strict Expert Precision: {metrics.strict_expert_precision}")
    print(f"  Breakdown: {metrics.validation_breakdown}")

    # Verify 10 Benchmark Research Gap Questions
    benchmark_suite = EvaluationService.get_benchmark_suite(db)
    print(f"\nMulti-Paper Research Gap Benchmark Suite ({len(benchmark_suite)} questions):")
    for b in benchmark_suite:
        print(f"  [{b['id']}] {b['category']}: {b['question']}")
    assert len(benchmark_suite) == 10, f"Expected 10 benchmark questions, got {len(benchmark_suite)}"
    assert len(metrics.benchmark_details) == 10, f"Expected 10 benchmark details, got {len(metrics.benchmark_details)}"

    # Baseline Comparison Test (Unrestricted User Question)
    comparison = EvaluationService.run_baseline_comparison("What research gaps exist in Byzantine-resilient optimization?")
    print("\nBaseline Comparison:")
    print(f"  Dense RAG Results Count: {len(comparison['dense_rag_results'])}")
    print(f"  Keyword Baseline Results Count: {len(comparison['plain_keyword_results'])}")

    print("[PASS] test_expert_validation_and_metrics PASSED")
    db.close()


if __name__ == "__main__":
    test_expert_validation_and_metrics()
