from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.schemas import ExpertFeedbackRequest, ExpertFeedbackResponse, EvaluationMetricsResponse
from app.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/evaluation", tags=["Evaluation & Validation Layer"])


@router.post("/expert-feedback", response_model=ExpertFeedbackResponse)
def submit_expert_feedback(req: ExpertFeedbackRequest, db: Session = Depends(get_db)):
    """Submit domain expert evaluation for a detected research gap:
    
    - 'genuine': The cluster represents a true, unexplored open research problem.
    - 'already_studied': The area has already been well-addressed in prior literature.
    - 'unclear': Insufficient evidence to determine.
    
    Returns updated system precision metric.
    """
    valid_ratings = {"genuine", "already_studied", "unclear"}
    if req.rating.lower().strip() not in valid_ratings:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid rating '{req.rating}'. Must be one of: {list(valid_ratings)}"
        )

    try:
        res = EvaluationService.submit_expert_feedback(db, req)
        return res
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=EvaluationMetricsResponse)
def get_evaluation_metrics(db: Session = Depends(get_db)):
    """Returns system evaluation metrics:
    - Total papers and chunks
    - Question Coverage@5 and Multi-Paper Evidence Rate
    - Answer Faithfulness Rate
    - Expert Validation Precision (genuine / total reviewed)
    - Full breakdown of expert judgments
    """
    return EvaluationService.get_evaluation_metrics(db)


@router.get("/benchmark-questions")
def get_benchmark_questions(db: Session = Depends(get_db)):
    """Returns the general academic benchmark questions used for retrieval evaluation."""
    return EvaluationService.get_benchmark_suite(db)


@router.post("/baseline-comparison")
def compare_with_baselines(question: str = Query(..., description="Query to compare between Dense RAG and Keyword Search")):
    """Compares Dense Semantic RAG retrieval vs Plain Keyword Search baseline."""
    return EvaluationService.run_baseline_comparison(question)
