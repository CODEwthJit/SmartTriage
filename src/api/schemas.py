"""
SmartTriage API Schemas.
Pydantic contracts for request validation, error enforcement,
and structured model inference responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class IssueRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=300, description="Title of the reported issue")
    body: Optional[str] = Field(default="", max_length=5000, description="Issue description or stack trace")

    model_config = {
        "json_schema_extra": {
            "example": {
                "title": "NullPointerException when cart checkout is empty",
                "body": "When submitting checkout without items, server crashes with 500 NPE in CheckoutService.",
            }
        }
    }


class DuplicateMatch(BaseModel):
    issue_id: int
    title: str
    category: str
    priority: str
    similarity_score: float
    is_duplicate_warning: bool


class TriageResponse(BaseModel):
    category: str = Field(..., description="Predicted issue component (e.g. bug, security, performance)")
    priority: str = Field(..., description="Estimated priority severity")
    confidence: float = Field(..., description="Model confidence score between 0.0 and 1.0")
    duplicate_warning: bool = Field(..., description="True if a high-similarity duplicate exists")
    top_duplicates: List[DuplicateMatch] = Field(default_factory=list, description="Top potential duplicate issues")
    latency_ms: float = Field(..., description="Inference processing latency in milliseconds")


class DuplicateSearchRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=300)
    body: Optional[str] = Field(default="", max_length=5000)
    top_k: int = Field(default=3, ge=1, le=10)
    threshold: float = Field(default=0.70, ge=0.0, le=1.0)


class DuplicateSearchResponse(BaseModel):
    query: str
    matches: List[DuplicateMatch]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    vector_index_loaded: bool
    indexed_documents: int
