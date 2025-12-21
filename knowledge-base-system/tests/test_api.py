"""
Tests for the API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert data["status"] == "running"


def test_health_endpoint():
    """Test the health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_list_companies():
    """Test listing companies."""
    response = client.get("/api/v1/companies")
    assert response.status_code == 200
    data = response.json()
    assert "companies" in data
    assert "total" in data
    assert isinstance(data["companies"], list)


def test_load_company():
    """Test loading a company."""
    response = client.post(
        "/api/v1/company/load",
        json={
            "company_id": "techstore",
            "force_reindex": False
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["company_id"] == "techstore"


def test_query_endpoint():
    """Test querying the knowledge base."""
    # First load a company
    client.post(
        "/api/v1/company/load",
        json={"company_id": "techstore", "force_reindex": False}
    )

    # Then query
    response = client.post(
        "/api/v1/query",
        json={
            "company_id": "techstore",
            "query": "What is your return policy?",
            "top_k": 5
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "chunks" in data
    assert "confidence_score" in data
    assert isinstance(data["chunks"], list)


def test_context_endpoint():
    """Test getting context for LLM."""
    # Load company first
    client.post(
        "/api/v1/company/load",
        json={"company_id": "techstore", "force_reindex": False}
    )

    # Get context
    response = client.post(
        "/api/v1/context",
        json={
            "company_id": "techstore",
            "query": "How can I track my order?",
            "top_k": 3,
            "include_metadata": True
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "formatted_context" in data
    assert "company_name" in data
    assert isinstance(data["formatted_context"], str)


def test_invalid_company():
    """Test querying with invalid company."""
    response = client.post(
        "/api/v1/query",
        json={
            "company_id": "nonexistent",
            "query": "test query",
            "top_k": 5
        }
    )
    assert response.status_code == 404


def test_statistics_endpoint():
    """Test the statistics endpoint."""
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
    assert "total_companies" in data
    assert "total_chunks" in data


def test_clear_cache():
    """Test clearing cache."""
    response = client.post("/api/v1/cache/clear")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
