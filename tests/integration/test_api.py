"""
Automated Integration Tests for SmartTriage FastAPI Service.
Verifies status codes, schema adherence, error handling, and latency constraints.
"""

from fastapi.testclient import TestClient
import pytest
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    """Verifies that /v1/health returns 200 OK and models are loaded in memory."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["classifier_loaded"] is True
    assert data["vector_index_loaded"] is True
    assert data["indexed_documents"] > 0


def test_triage_bug_issue(client):
    """Verifies triage endpoint correctly classifies a bug and detects NPE duplicate."""
    payload = {
        "title": "NullPointerException when cart checkout is empty",
        "body": "Submitting checkout without items causes 500 error due to NPE in CheckoutService.",
    }
    response = client.post("/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "bug"
    assert "priority" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["latency_ms"] < 250.0  # Latency contract test: must be sub-250ms
    assert len(data["top_duplicates"]) > 0


def test_triage_security_issue(client):
    """Verifies triage endpoint correctly flags security vulnerability."""
    payload = {
        "title": "SQL injection vulnerability in user search filter",
        "body": "User filter parameter is concatenated into raw SQL string.",
    }
    response = client.post("/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "security"
    assert data["priority"] == "P0-Critical"


def test_validation_error_on_short_title(client):
    """Pydantic must reject titles shorter than 3 characters with 422 Unprocessable Entity."""
    payload = {"title": "ab"}
    response = client.post("/v1/triage", json=payload)
    assert response.status_code == 422
