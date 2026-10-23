import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.db_models import Paper, GapAnalysisRun, ClusterRecord
from app.models.schemas import GapAnalysisRequest, GapAnalysisRunResponse, ClusterGapResponse, EvidenceQuote
from app.services.gap_analyzer import GapAnalyzer

router = APIRouter(prefix="/gaps", tags=["Research Gap Identification Layer"])


@router.post("/analyze", response_model=GapAnalysisRunResponse)
def trigger_gap_analysis(req: GapAnalysisRequest = GapAnalysisRequest(), db: Session = Depends(get_db)):
    """Triggers unsupervised paper clustering and computes the 4 research gap signals:
    
    1. Semantic Sparsity: Isolation and small representation in embedding space.
    2. Temporal Decay: Momentum dropped or halted in recent years.
    3. Contradiction Density: Disagreements and conflicting findings in cluster papers.
    4. Explicit Author Admission: Direct quotes from limitations and future-work sections.
    
    Combined into a normalized Gap Score: w1*sparsity + w2*decay + w3*contradiction + w4*admission.
    """
    total_papers = db.query(Paper).count()
    if total_papers < 2:
        raise HTTPException(
            status_code=400,
            detail=f"At least 2 papers are required to run cluster gap analysis. Current count: {total_papers}. Please ingest papers first."
        )

    try:
        run_response = GapAnalyzer.run_analysis(
            db=db,
            weights=req.weights,
            num_clusters=req.num_clusters
        )
        return run_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gap analysis error: {str(e)}")


@router.get("/clusters", response_model=List[ClusterGapResponse])
def get_latest_gap_clusters(db: Session = Depends(get_db)):
    """Fetch the latest ranked research gap clusters with signal scores, traceable evidence quotes, and 2D coordinates."""
    latest_run = db.query(GapAnalysisRun).order_by(GapAnalysisRun.created_at.desc()).first()
    if not latest_run:
        raise HTTPException(status_code=404, detail="No gap analysis has been performed yet. Run /api/gaps/analyze first.")

    clusters = db.query(ClusterRecord).filter(ClusterRecord.run_id == latest_run.run_id).order_by(ClusterRecord.gap_score.desc()).all()

    results = []
    for c in clusters:
        ev_quotes = [
            EvidenceQuote(**eq)
            for eq in json.loads(c.evidence_quotes_json)
        ] if c.evidence_quotes_json else []

        paper_ids = json.loads(c.paper_ids_json) if c.paper_ids_json else []
        paper_titles = json.loads(c.paper_titles_json) if c.paper_titles_json else []

        results.append(
            ClusterGapResponse(
                cluster_id=c.id,
                cluster_label=c.cluster_label,
                topic_title=c.topic_title,
                paper_ids=paper_ids,
                paper_titles=paper_titles,
                paper_count=len(paper_ids),
                sparsity_score=c.sparsity_score,
                decay_score=c.decay_score,
                contradiction_score=c.contradiction_score,
                admission_score=c.admission_score,
                gap_score=c.gap_score,
                evidence_quotes=ev_quotes,
                suggested_directions=c.suggested_directions or "",
                pca_x=c.pca_x,
                pca_y=c.pca_y
            )
        )
    return results


@router.get("/report/{cluster_id}")
def generate_gap_report(cluster_id: int, format: str = "json", db: Session = Depends(get_db)):
    """Generate an explainable, evidence-backed report for a detected research gap."""
    cluster = db.query(ClusterRecord).filter(ClusterRecord.id == cluster_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster record not found.")

    paper_titles = json.loads(cluster.paper_titles_json) if cluster.paper_titles_json else []
    evidence_quotes = json.loads(cluster.evidence_quotes_json) if cluster.evidence_quotes_json else []

    if format.lower() == "markdown":
        md = f"""# Research Gap Report: {cluster.topic_title}
**Overall Gap Score:** {cluster.gap_score:.3f} / 1.000

---

## 1. Signal Breakdown
- **Semantic Sparsity:** {cluster.sparsity_score:.3f} (Measures topic isolation & under-representation)
- **Temporal Decay:** {cluster.decay_score:.3f} (Measures publication momentum decline)
- **Contradiction Density:** {cluster.contradiction_score:.3f} (Measures conflicting findings & empirical tension)
- **Explicit Author Admission:** {cluster.admission_score:.3f} (Measures author admissions in limitations/future work)

---

## 2. Traceable Evidence from Literature
"""
        if evidence_quotes:
            for eq in evidence_quotes:
                md += f"- **[{eq.get('paper_title')}, {eq.get('year')}] ({eq.get('section_name')} - {eq.get('signal_type')}):**\n"
                md += f"  > \"{eq.get('quote')}\"\n\n"
        else:
            md += "No explicit admission sentences captured in current chunks.\n\n"

        md += f"""## 3. Papers Involved ({len(paper_titles)})
"""
        for pt in paper_titles:
            md += f"- {pt}\n"

        md += f"""
---

## 4. Suggested Future Research Direction
{cluster.suggested_directions}
"""
        return PlainTextResponse(content=md, media_type="text/markdown")

    return {
        "cluster_id": cluster.id,
        "topic_title": cluster.topic_title,
        "overall_gap_score": cluster.gap_score,
        "signals": {
            "semantic_sparsity": cluster.sparsity_score,
            "temporal_decay": cluster.decay_score,
            "contradiction_density": cluster.contradiction_score,
            "explicit_author_admission": cluster.admission_score
        },
        "papers_involved": paper_titles,
        "evidence_quotes": evidence_quotes,
        "suggested_future_direction": cluster.suggested_directions,
        "map_coordinates": {"x": cluster.pca_x, "y": cluster.pca_y}
    }
