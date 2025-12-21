"""
Pydantic models for request/response validation and data structures.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import numpy as np


class DocumentType(str, Enum):
    """Types of documents in the knowledge base."""
    FAQ = "faq"
    PRODUCT = "product"
    POLICY = "policy"
    GENERAL = "general"


class Priority(int, Enum):
    """Priority levels for chunks."""
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class ChunkMetadata(BaseModel):
    """Metadata attached to each document chunk."""
    chunk_id: str
    company_id: str
    document_id: str
    document_type: DocumentType
    category: str
    chunk_index: int
    total_chunks: int
    source_file: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    priority: Priority = Priority.MEDIUM

    class Config:
        use_enum_values = True


class DocumentChunk(BaseModel):
    """Represents a chunk of text with metadata and optional embedding."""
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    embedding: Optional[Any] = None  # np.ndarray, but Pydantic doesn't handle it well

    class Config:
        arbitrary_types_allowed = True

    def dict(self, **kwargs):
        """Custom dict method to handle numpy arrays."""
        d = super().dict(**kwargs)
        if self.embedding is not None and isinstance(self.embedding, np.ndarray):
            d['embedding'] = self.embedding.tolist()
        return d


class CompanyMetadata(BaseModel):
    """Metadata about a company in the system."""
    company_id: str
    company_name: str
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    total_documents: int = 0
    total_chunks: int = 0
    categories: List[str] = []


class QueryRequest(BaseModel):
    """Request model for querying the knowledge base."""
    company_id: str
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = None

    @validator('query')
    def validate_query(cls, v):
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        return v.strip()


class RetrievedChunk(BaseModel):
    """A chunk retrieved from the knowledge base with its similarity score."""
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    similarity_score: float
    source: str


class QueryResponse(BaseModel):
    """Response model for knowledge base queries."""
    chunks: List[RetrievedChunk]
    confidence_score: float
    sources: List[str]
    retrieval_time: float
    company_id: str
    total_results: int


class ContextRequest(BaseModel):
    """Request for formatted context for LLM."""
    company_id: str
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = None
    include_metadata: bool = True


class ContextResponse(BaseModel):
    """Formatted context ready for LLM consumption."""
    formatted_context: str
    company_name: str
    query: str
    num_chunks: int
    confidence_score: float
    retrieval_time: float


class CompanySwitchRequest(BaseModel):
    """Request to switch active company."""
    company_id: str


class CompanySwitchResponse(BaseModel):
    """Response after switching company."""
    success: bool
    company_id: str
    company_name: str
    message: str
    total_chunks: int


class IngestRequest(BaseModel):
    """Request to ingest/re-ingest company documents."""
    company_id: str
    force_reindex: bool = False


class IngestResponse(BaseModel):
    """Response after ingesting documents."""
    success: bool
    company_id: str
    chunks_processed: int
    documents_processed: int
    processing_time: float
    message: str


class HealthResponse(BaseModel):
    """System health status response."""
    status: str
    current_company: Optional[str]
    total_chunks: int
    memory_usage_mb: float
    uptime_seconds: float
    companies_loaded: int
    embedding_model: str


class StatisticsResponse(BaseModel):
    """Usage statistics response."""
    total_queries: int
    total_companies: int
    total_chunks: int
    average_query_time: float
    cache_hit_rate: float
    uptime_seconds: float
    queries_per_company: Dict[str, int]


class CompanyListResponse(BaseModel):
    """List of available companies."""
    companies: List[CompanyMetadata]
    total: int


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
