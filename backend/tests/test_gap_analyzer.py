import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.session import SessionLocal
from app.services.gap_analyzer import GapAnalyzer
from app.models.schemas import GapWeights


def test_gap_analysis():
    db = SessionLocal()
    print("\n--- Running Research Gap Analysis Across Corpus ---")
    weights = GapWeights(
        w_sparsity=0.25,
        w_decay=0.25,
        w_contradiction=0.25,
        w_admission=0.25
    )

    result = GapAnalyzer.run_analysis(db=db, weights=weights, num_clusters=3)

    print(f"Total Papers Analyzed: {result.total_papers}")
    print(f"Total Clusters Formed: {result.num_clusters}")
    print(f"Weights Applied: {result.weights}\n")

    for i, cluster in enumerate(result.ranked_gaps):
        print(f"Rank {i+1}: Cluster '{cluster.topic_title}' (Papers: {cluster.paper_count})")
        print(f"  Overall Gap Score: {cluster.gap_score:.3f}")
        print(f"  - Semantic Sparsity:     {cluster.sparsity_score:.3f}")
        print(f"  - Temporal Decay:        {cluster.decay_score:.3f}")
        print(f"  - Contradiction Density: {cluster.contradiction_score:.3f}")
        print(f"  - Author Admission:      {cluster.admission_score:.3f}")
        print(f"  2D Coordinates: ({cluster.pca_x}, {cluster.pca_y})")
        print(f"  Evidence Quotes Count: {len(cluster.evidence_quotes)}")
        for eq in cluster.evidence_quotes[:2]:
            print(f"    * [{eq.paper_title}] ({eq.section_name} - {eq.signal_type}): \"{eq.quote[:100]}...\"")
        safe_dir = cluster.suggested_directions.encode("ascii", "replace").decode("ascii")
        print(f"  Suggested Direction: {safe_dir}\n")

        assert 0.0 <= cluster.gap_score <= 1.0, "Gap score outside [0, 1]"
        assert 0.0 <= cluster.sparsity_score <= 1.0, "Sparsity score outside [0, 1]"
        assert 0.0 <= cluster.decay_score <= 1.0, "Decay score outside [0, 1]"
        assert 0.0 <= cluster.contradiction_score <= 1.0, "Contradiction score outside [0, 1]"
        assert 0.0 <= cluster.admission_score <= 1.0, "Admission score outside [0, 1]"

    assert len(result.ranked_gaps) > 0, "No clusters returned"
    print("[PASS] test_gap_analysis PASSED")
    db.close()


if __name__ == "__main__":
    test_gap_analysis()
