"""Pydantic request/response schemas for the API layer."""
from typing import List, Optional

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    intent: Optional[str] = None
    query_type: Optional[str] = None


class UploadResponse(BaseModel):
    message: str
    file_id: str
    chunks_indexed: int


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    vectorstore_ready: bool
    openai_configured: bool


class LatencyStats(BaseModel):
    count: int = 0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0


class SimilarityStats(BaseModel):
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0


class AnalyticsSummaryResponse(BaseModel):
    total_queries: int
    answer_rate: float
    latency: dict
    similarity: dict
    precision_at_k_proxy: float


class StatsResponse(BaseModel):
    total_documents_indexed: int
    total_chunks: int
    vector_db_size_bytes: int
    avg_tokens_per_query: float
    estimated_cost_per_query_usd: float
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    cache_size: int = 0
    cache_maxsize: int = 0
