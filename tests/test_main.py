"""Tests for Hindsight API endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac




@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_create_document(client: AsyncClient):
    response = await client.post(
        "/v1/banks/test-bank/documents",
        json={"id": "doc-1", "content": "Test content", "metadata": {"key": "value"}}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["id"] == "doc-1"
    assert data["content"] == "Test content"
    assert data["metadata"] == {"key": "value"}
    assert data["bank_id"] == "test-bank"


@pytest.mark.asyncio
async def test_get_document(client: AsyncClient):
    # First create a document
    await client.post(
        "/v1/banks/test-bank/documents",
        json={"id": "doc-2", "content": "Another content", "metadata": {}}
    )
    # Then retrieve it
    response = await client.get("/v1/banks/test-bank/documents/doc-2")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "doc-2"
    assert data["content"] == "Another content"


@pytest.mark.asyncio
async def test_get_nonexistent_document(client: AsyncClient):
    response = await client.get("/v1/banks/test-bank/documents/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_recall_memories(client: AsyncClient):
    # Add some memories first via the store (since we don't have a retain endpoint yet)
    from app.main import store
    store.add_memory("test-bank", {"id": "mem-1", "text": "User likes coffee", "mentioned_at": "2024-01-15T10:00:00Z"})
    store.add_memory("test-bank", {"id": "mem-2", "text": "User prefers tea", "mentioned_at": "2024-01-16T10:00:00Z"})
    store.add_memory("other-bank", {"id": "mem-3", "text": "Unrelated memory"})

    response = await client.post(
        "/v1/banks/test-bank/recall",
        json={"query": "coffee", "limit": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["text"] == "User likes coffee"
    assert data["results"][0]["mentioned_at"] == "2024-01-15T10:00:00Z"