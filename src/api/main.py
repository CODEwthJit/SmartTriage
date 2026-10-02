"""
SmartTriage Production FastAPI Gateway.
Serves real-time issue classification and semantic duplicate detection
with sub-millisecond singleton models in memory and strict Pydantic contracts.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import time
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import joblib

from src.api.schemas import (
    IssueRequest,
    TriageResponse,
    DuplicateMatch,
    DuplicateSearchRequest,
    DuplicateSearchResponse,
    HealthResponse,
)
from src.inference.vector_search import DuplicateSearchEngine

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_REGISTRY_DIR = PROJECT_ROOT / "models" / "registry"
CLASSIFIER_PATH = MODEL_REGISTRY_DIR / "baseline_pipeline.joblib"
INDEX_PATH = MODEL_REGISTRY_DIR / "vector_index.joblib"

# Priority heuristic map for baseline fallback
CATEGORY_PRIORITY_MAP = {
    "security": "P0-Critical",
    "bug": "P1-High",
    "performance": "P1-High",
    "feature": "P2-Medium",
    "documentation": "P3-Low",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    MLOps Singleton Pattern:
    Loads machine learning models once into RAM during startup.
    Avoids re-reading heavy disk binaries on every incoming HTTP request.
    """
    print("--- [STARTUP] Initializing SmartTriage Model Engine ---")
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Classifier model missing at: {CLASSIFIER_PATH}. Run baseline.py first.")
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Vector index missing at: {INDEX_PATH}. Run vector_search.py first.")

    # 1. Load trained classification pipeline
    print(f"Loading classifier from {CLASSIFIER_PATH}...")
    app.state.classifier = joblib.load(CLASSIFIER_PATH)

    # 2. Load semantic vector search engine
    print(f"Loading vector search engine and precomputed embeddings from {INDEX_PATH}...")
    search_engine = DuplicateSearchEngine()
    search_engine.load(INDEX_PATH)
    app.state.search_engine = search_engine

    print("--- [STARTUP COMPLETE] SmartTriage API is Ready for Live Traffic ---")
    yield
    print("--- [SHUTDOWN] Unloading models and freeing memory ---")


app = FastAPI(
    title="SmartTriage API",
    version="1.0.0",
    description="Production AI/ML Service for Intelligent GitHub Issue Triage & Semantic Deduplication",
    lifespan=lifespan,
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Liveness & readiness probe verifying model states."""
    classifier_ready = hasattr(app.state, "classifier") and app.state.classifier is not None
    engine_ready = hasattr(app.state, "search_engine") and app.state.search_engine is not None
    doc_count = len(app.state.search_engine.metadata) if engine_ready else 0

    return HealthResponse(
        status="healthy" if classifier_ready and engine_ready else "degraded",
        classifier_loaded=classifier_ready,
        vector_index_loaded=engine_ready,
        indexed_documents=doc_count,
    )


@app.post("/v1/triage", response_model=TriageResponse, tags=["Inference"])
async def triage_issue(request: IssueRequest):
    """
    Full Issue Triage Pipeline:
    1. Predicts component category (e.g. security, bug, performance) with calibrated confidence.
    2. Estimates priority severity.
    3. Searches for semantic duplicates in historical index with duplicate warnings.
    """
    start_time = time.perf_counter()
    full_text = f"{request.title} {request.body}".strip()

    try:
        # Step 1: Classification & Probability Calibration
        pred_category = app.state.classifier.predict([full_text])[0]
        probabilities = app.state.classifier.predict_proba([full_text])[0]
        confidence = float(max(probabilities))

        # Priority estimation heuristic based on predicted class and urgency keywords
        pred_priority = CATEGORY_PRIORITY_MAP.get(pred_category, "P2-Medium")
        if any(w in full_text.lower() for w in ["critical", "crash", "blocker", "outage", "deadlock"]):
            pred_priority = "P0-Critical"

        # Step 2: Semantic Duplicate Search (top 3 candidates)
        raw_matches = app.state.search_engine.query(title=request.title, body=request.body, top_k=3, threshold=0.70)

        duplicates = [DuplicateMatch(**m) for m in raw_matches]
        has_duplicate_warning = any(m.is_duplicate_warning for m in duplicates)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return TriageResponse(
            category=pred_category,
            priority=pred_priority,
            confidence=round(confidence, 4),
            duplicate_warning=has_duplicate_warning,
            top_duplicates=duplicates,
            latency_ms=latency_ms,
        )

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}")


@app.post("/v1/duplicates", response_model=DuplicateSearchResponse, tags=["Inference"])
async def search_duplicates(request: DuplicateSearchRequest):
    """Dedicated endpoint for semantic duplicate issue retrieval."""
    start_time = time.perf_counter()
    try:
        raw_matches = app.state.search_engine.query(
            title=request.title, body=request.body, top_k=request.top_k, threshold=request.threshold
        )
        duplicates = [DuplicateMatch(**m) for m in raw_matches]
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return DuplicateSearchResponse(
            query=f"{request.title} {request.body}".strip(), matches=duplicates, latency_ms=latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Vector search error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
