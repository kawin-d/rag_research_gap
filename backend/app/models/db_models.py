from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database.session import Base


class Paper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(String(128), unique=True, index=True, nullable=False)
    title = Column(String(512), nullable=False)
    authors = Column(Text, default="[]")  # JSON list of author strings
    year = Column(Integer, nullable=True, index=True)
    abstract = Column(Text, default="")
    citation_count = Column(Integer, default=0)
    url = Column(String(512), default="")
    pdf_path = Column(String(512), default="")
    sections_json = Column(Text, default="{}")  # JSON dict of parsed sections
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    chunks = relationship("Chunk", back_populates="paper", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)
    paper_uid = Column(String(128), index=True, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    section_name = Column(String(128), index=True, default="general")
    text = Column(Text, nullable=False)
    token_count = Column(Integer, default=0)
    embedding_index = Column(Integer, default=-1)

    paper = relationship("Paper", back_populates="chunks")


class GapAnalysisRun(Base):
    __tablename__ = "gap_analysis_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(64), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    weights_json = Column(Text, default="{}")
    total_papers = Column(Integer, default=0)
    num_clusters = Column(Integer, default=0)

    clusters = relationship("ClusterRecord", back_populates="run", cascade="all, delete-orphan")


class ClusterRecord(Base):
    __tablename__ = "cluster_records"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(64), ForeignKey("gap_analysis_runs.run_id", ondelete="CASCADE"), nullable=False, index=True)
    cluster_label = Column(Integer, nullable=False)
    topic_title = Column(String(256), nullable=False)
    paper_ids_json = Column(Text, default="[]")
    paper_titles_json = Column(Text, default="[]")
    sparsity_score = Column(Float, default=0.0)
    decay_score = Column(Float, default=0.0)
    contradiction_score = Column(Float, default=0.0)
    admission_score = Column(Float, default=0.0)
    gap_score = Column(Float, default=0.0)
    evidence_quotes_json = Column(Text, default="[]")
    suggested_directions = Column(Text, default="")
    pca_x = Column(Float, default=0.0)
    pca_y = Column(Float, default=0.0)

    run = relationship("GapAnalysisRun", back_populates="clusters")
    validations = relationship("ExpertValidation", back_populates="cluster", cascade="all, delete-orphan")


class ExpertValidation(Base):
    __tablename__ = "expert_validations"

    id = Column(Integer, primary_key=True, index=True)
    cluster_record_id = Column(Integer, ForeignKey("cluster_records.id", ondelete="CASCADE"), nullable=False, index=True)
    expert_name = Column(String(128), default="Domain Expert")
    rating = Column(String(64), nullable=False)  # 'genuine', 'already_studied', 'unclear'
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    cluster = relationship("ClusterRecord", back_populates="validations")


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    citations_json = Column(Text, default="[]")
    retrieved_chunks_json = Column(Text, default="[]")
    created_at = Column(DateTime, default=datetime.utcnow)
