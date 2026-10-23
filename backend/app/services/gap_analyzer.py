import re
import uuid
import json
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy.orm import Session

from app.config import settings
from app.models.db_models import Paper, Chunk, GapAnalysisRun, ClusterRecord
from app.models.schemas import ClusterGapResponse, EvidenceQuote, GapAnalysisRunResponse, GapWeights
from app.services.embedding_service import EmbeddingService


# Author admission regex cues
ADMISSION_PATTERNS = [
    re.compile(r"([^.!?]*\b(?:open (?:question|problem|challenge)|leave (?:this|it)? for future work|limitation of (?:our|this)|remains unexplored|unexplored area|future research should|further investigation is needed|does not scale|lack of|not yet understood|unaddressed)[^.!?]*[.!?])", re.IGNORECASE),
    re.compile(r"([^.!?]*\b(?:we have not investigated|beyond the scope of this paper|remains unclear|promising avenue for future work)[^.!?]*[.!?])", re.IGNORECASE)
]

# Contradiction and debate cues
CONTRADICTION_PATTERNS = [
    re.compile(r"([^.!?]*\b(?:however|in contrast|contrary to|conflicting results|contradicts|failed to replicate|discrepancy|disagreement|contested|trade-off between|degrades under)[^.!?]*[.!?])", re.IGNORECASE),
    re.compile(r"([^.!?]*\b(?:mixed results|still debating|inconsistent with|sub-optimal under)[^.!?]*[.!?])", re.IGNORECASE)
]


class GapAnalyzer:
    """Computes paper topic clusters and calculates the 4 Research Gap Signals."""

    @classmethod
    def run_analysis(
        cls,
        db: Session,
        weights: Optional[GapWeights] = None,
        num_clusters: Optional[int] = None
    ) -> GapAnalysisRunResponse:
        # Load all papers and their chunks
        papers = db.query(Paper).all()
        if len(papers) < 2:
            raise ValueError(f"At least 2 papers are required to run cluster gap analysis. Currently have: {len(papers)}")

        w1 = weights.w_sparsity if weights else settings.WEIGHT_SPARSITY
        w2 = weights.w_decay if weights else settings.WEIGHT_DECAY
        w3 = weights.w_contradiction if weights else settings.WEIGHT_CONTRADICTION
        w4 = weights.w_admission if weights else settings.WEIGHT_ADMISSION

        # Normalize weights so sum is 1.0
        w_sum = w1 + w2 + w3 + w4
        if w_sum > 0:
            w1, w2, w3, w4 = w1 / w_sum, w2 / w_sum, w3 / w_sum, w4 / w_sum

        # 1. Embed paper abstracts / titles for clustering
        paper_texts = [f"{p.title}. {p.abstract or ''}" for p in papers]
        paper_embeddings = EmbeddingService.embed_texts(paper_texts)

        # 2. Determine cluster count
        n_clusters = num_clusters
        if not n_clusters:
            # Heuristic: k between 2 and 6 depending on paper count
            n_clusters = max(2, min(6, len(papers) // 3 if len(papers) >= 6 else 2))
        n_clusters = min(n_clusters, len(papers))

        # Run clustering (KMeans)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(paper_embeddings)

        # 3. 2D Map Coordinates (PCA)
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(paper_embeddings)

        # 4. Group papers by cluster
        clusters_data = {i: {"papers": [], "coords": []} for i in range(n_clusters)}
        for idx, (p, label) in enumerate(zip(papers, labels)):
            clusters_data[label]["papers"].append(p)
            clusters_data[label]["coords"].append(coords_2d[idx])

        # Global centroid for sparsity computation
        global_centroid = np.mean(paper_embeddings, axis=0)

        run_id = f"run_{uuid.uuid4().hex[:10]}"
        ranked_gaps: List[ClusterGapResponse] = []

        current_year = datetime.utcnow().year

        for cluster_idx, c_info in clusters_data.items():
            c_papers: List[Paper] = c_info["papers"]
            if not c_papers:
                continue

            c_paper_ids = [p.paper_id for p in c_papers]
            c_titles = [p.title for p in c_papers]
            c_years = [p.year for p in c_papers if p.year is not None]

            # Fetch all chunks for this cluster's papers
            c_db_ids = [p.id for p in c_papers]
            cluster_chunks = db.query(Chunk).filter(Chunk.paper_id.in_(c_db_ids)).all()

            # --- Extract Topic Title (TF-IDF Top Keywords) ---
            cluster_corpus = [f"{p.title} {p.abstract}" for p in c_papers]
            topic_title = cls._extract_topic_title(cluster_corpus, cluster_idx)

            # --- Signal 1: Semantic Sparsity ---
            # Measures: cluster size relative to total + distance from global embedding mass
            size_sparsity = 1.0 - (len(c_papers) / len(papers))
            c_vectors = paper_embeddings[[i for i, l in enumerate(labels) if l == cluster_idx]]
            c_centroid = np.mean(c_vectors, axis=0)
            isolation_dist = float(np.linalg.norm(c_centroid - global_centroid))
            # Normalization factor
            sparsity_score = float(np.clip(0.6 * size_sparsity + 0.4 * min(isolation_dist, 1.0), 0.0, 1.0))

            # --- Signal 2: Temporal Decay ---
            # Measures if publications peaked in past years and slowed down/stalled recently
            if c_years:
                recent_cutoff = current_year - 2
                recent_papers = sum(1 for y in c_years if y >= recent_cutoff)
                older_papers = sum(1 for y in c_years if y < recent_cutoff)
                if older_papers > 0 and recent_papers == 0:
                    decay_score = 0.90  # Stopped having publications recently!
                elif older_papers > recent_papers:
                    decay_score = 0.70  # Declining pace
                elif recent_papers > 0 and older_papers == 0:
                    decay_score = 0.20  # Brand new emerging, not decaying
                else:
                    decay_score = 0.45  # Steady
            else:
                decay_score = 0.50

            # --- Signal 3: Contradiction Density ---
            # Scans chunks for conflicting assertions, disagreements, or empirical tension
            contradiction_quotes, raw_contradiction_score = cls._find_contradictions(cluster_chunks, c_papers)
            contradiction_score = float(np.clip(raw_contradiction_score, 0.0, 1.0))

            # --- Signal 4: Explicit Author Admission ---
            # Scans limitations, future work, and conclusion chunks for open gap admissions
            admission_quotes, raw_admission_score = cls._find_author_admissions(cluster_chunks, c_papers)
            admission_score = float(np.clip(raw_admission_score, 0.0, 1.0))

            # Combined Gap Score
            gap_score = float(round(
                w1 * sparsity_score +
                w2 * decay_score +
                w3 * contradiction_score +
                w4 * admission_score,
                4
            ))

            # Generate suggested future direction
            suggested_direction = cls._synthesize_future_direction(
                topic_title, admission_quotes, contradiction_quotes
            )

            # Cluster average coordinates on 2D map
            coords_arr = np.array(c_info["coords"])
            mean_x = float(coords_arr[:, 0].mean()) if len(coords_arr) > 0 else 0.0
            mean_y = float(coords_arr[:, 1].mean()) if len(coords_arr) > 0 else 0.0

            all_evidence = admission_quotes[:3] + contradiction_quotes[:2]

            ranked_gaps.append(
                ClusterGapResponse(
                    cluster_id=cluster_idx,
                    cluster_label=cluster_idx,
                    topic_title=topic_title,
                    paper_ids=c_paper_ids,
                    paper_titles=c_titles,
                    paper_count=len(c_papers),
                    sparsity_score=round(sparsity_score, 3),
                    decay_score=round(decay_score, 3),
                    contradiction_score=round(contradiction_score, 3),
                    admission_score=round(admission_score, 3),
                    gap_score=round(gap_score, 3),
                    evidence_quotes=all_evidence,
                    suggested_directions=suggested_direction,
                    pca_x=round(mean_x, 4),
                    pca_y=round(mean_y, 4)
                )
            )

        # Sort clusters descending by Gap Score
        ranked_gaps.sort(key=lambda x: x.gap_score, reverse=True)

        # Save to database
        db_run = GapAnalysisRun(
            run_id=run_id,
            weights_json=json.dumps({"w1": w1, "w2": w2, "w3": w3, "w4": w4}),
            total_papers=len(papers),
            num_clusters=len(ranked_gaps)
        )
        db.add(db_run)
        db.flush()

        for g in ranked_gaps:
            rec = ClusterRecord(
                run_id=run_id,
                cluster_label=g.cluster_label,
                topic_title=g.topic_title,
                paper_ids_json=json.dumps(g.paper_ids),
                paper_titles_json=json.dumps(g.paper_titles),
                sparsity_score=g.sparsity_score,
                decay_score=g.decay_score,
                contradiction_score=g.contradiction_score,
                admission_score=g.admission_score,
                gap_score=g.gap_score,
                evidence_quotes_json=json.dumps([e.dict() for e in g.evidence_quotes]),
                suggested_directions=g.suggested_directions,
                pca_x=g.pca_x,
                pca_y=g.pca_y
            )
            db.add(rec)
            db.flush()
            g.cluster_id = rec.id  # Update response with primary key for expert feedback!

        db.commit()

        return GapAnalysisRunResponse(
            run_id=run_id,
            created_at=datetime.utcnow(),
            total_papers=len(papers),
            num_clusters=len(ranked_gaps),
            weights={"sparsity": w1, "decay": w2, "contradiction": w3, "admission": w4},
            ranked_gaps=ranked_gaps
        )

    @classmethod
    def _extract_topic_title(cls, corpus: List[str], cluster_idx: int) -> str:
        """Extract top representative keywords using TF-IDF."""
        try:
            tfidf = TfidfVectorizer(stop_words="english", max_features=10, ngram_range=(1, 2))
            X = tfidf.fit_transform(corpus)
            feature_names = tfidf.get_feature_names_out()
            scores = np.asarray(X.sum(axis=0)).flatten()
            top_indices = scores.argsort()[::-1][:3]
            keywords = [feature_names[i].title() for i in top_indices]
            return " / ".join(keywords) if keywords else f"Research Cluster {cluster_idx + 1}"
        except Exception:
            return f"Topic Cluster {cluster_idx + 1}"

    @classmethod
    def _find_author_admissions(cls, chunks: List[Chunk], papers: List[Paper]) -> Tuple[List[EvidenceQuote], float]:
        """Scans limitations, future work, and conclusion chunks for direct admission quotes."""
        paper_map = {p.id: p for p in papers}
        quotes: List[EvidenceQuote] = []
        admission_hits = 0

        # Prioritize relevant sections
        target_sections = {"limitations_future_work", "conclusion", "results", "general"}

        for c in chunks:
            if c.section_name in target_sections:
                p = paper_map.get(c.paper_id)
                p_title = p.title if p else "Paper"
                p_uid = p.paper_id if p else ""
                p_year = p.year if p else None

                for pat in ADMISSION_PATTERNS:
                    matches = pat.findall(c.text)
                    for m in matches:
                        clean_quote = m.strip()
                        if 25 < len(clean_quote) < 300:
                            quotes.append(
                                EvidenceQuote(
                                    paper_id=p_uid,
                                    paper_title=p_title,
                                    year=p_year,
                                    section_name=c.section_name,
                                    quote=clean_quote,
                                    signal_type="admission"
                                )
                            )
                            admission_hits += 1

        # Calculate signal score: more independent admissions -> higher score
        score = min(1.0, 0.25 * admission_hits) if admission_hits > 0 else 0.15
        return quotes, score

    @classmethod
    def _find_contradictions(cls, chunks: List[Chunk], papers: List[Paper]) -> Tuple[List[EvidenceQuote], float]:
        """Scans chunks for debate, tension, and contradiction indicators."""
        paper_map = {p.id: p for p in papers}
        quotes: List[EvidenceQuote] = []
        tension_hits = 0

        for c in chunks:
            p = paper_map.get(c.paper_id)
            p_title = p.title if p else "Paper"
            p_uid = p.paper_id if p else ""
            p_year = p.year if p else None

            for pat in CONTRADICTION_PATTERNS:
                matches = pat.findall(c.text)
                for m in matches:
                    clean_quote = m.strip()
                    if 25 < len(clean_quote) < 300:
                        quotes.append(
                            EvidenceQuote(
                                paper_id=p_uid,
                                paper_title=p_title,
                                year=p_year,
                                section_name=c.section_name,
                                quote=clean_quote,
                                signal_type="contradiction"
                            )
                        )
                        tension_hits += 1

        score = min(1.0, 0.20 * tension_hits) if tension_hits > 0 else 0.10
        return quotes, score

    @classmethod
    def _synthesize_future_direction(
        cls,
        topic: str,
        admissions: List[EvidenceQuote],
        contradictions: List[EvidenceQuote]
    ) -> str:
        """Generates an actionable research hypothesis / future work proposal based on evidence."""
        points = []
        if admissions:
            first_adm = admissions[0].quote
            points.append(f"Investigate unaddressed limitations identified in recent studies: \"{first_adm}\"")
        if contradictions:
            first_con = contradictions[0].quote
            points.append(f"Resolve empirical discrepancies regarding: \"{first_con}\"")

        if not points:
            return f"Conduct systematic comparative evaluation across diverse benchmark domains for {topic}."

        return " Furthermore, ".join(points)
