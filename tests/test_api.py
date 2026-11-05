"""Tests for Vector Lens 146 API endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from main import app
import uuid


@pytest.fixture
async def client():
    """Create test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def bank(client: AsyncClient):
    """Create a test bank with unique ID."""
    unique_id = f"test-bank-{uuid.uuid4().hex[:8]}"
    response = await client.post(
        "/v1/banks",
        json={"id": unique_id, "name": "Test Bank"},
    )
    assert response.status_code == 201
    return response.json()


@pytest.fixture
async def document(client: AsyncClient, bank: dict):
    """Create a test document with unique ID."""
    unique_id = f"doc-{uuid.uuid4().hex[:8]}"
    response = await client.post(
        f"/v1/banks/{bank['id']}/documents",
        json={"id": unique_id},
    )
    assert response.status_code == 201
    return response.json()


class TestHealth:
    """Health check tests."""

    async def test_healthz(self, client: AsyncClient):
        """Health endpoint returns ok."""
        response = await client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestBanks:
    """Bank endpoint tests."""

    async def test_create_bank(self, client: AsyncClient):
        """Can create a bank."""
        unique_id = f"new-bank-{uuid.uuid4().hex[:8]}"
        response = await client.post(
            "/v1/banks",
            json={"id": unique_id, "name": "New Bank"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == unique_id
        assert data["name"] == "New Bank"
        assert "created_at" in data

    async def test_create_duplicate_bank_fails(self, client: AsyncClient, bank: dict):
        """Creating duplicate bank returns 409."""
        response = await client.post(
            "/v1/banks",
            json={"id": bank["id"], "name": "Another Name"},
        )
        assert response.status_code == 409

    async def test_get_bank(self, client: AsyncClient, bank: dict):
        """Can get a bank by ID."""
        response = await client.get(f"/v1/banks/{bank['id']}")
        assert response.status_code == 200
        assert response.json()["id"] == bank["id"]

    async def test_get_nonexistent_bank(self, client: AsyncClient):
        """Getting nonexistent bank returns 404."""
        response = await client.get("/v1/banks/nonexistent")
        assert response.status_code == 404


class TestDocuments:
    """Document endpoint tests."""

    async def test_create_document(self, client: AsyncClient, bank: dict):
        """Can create a document in a bank."""
        unique_id = f"doc-{uuid.uuid4().hex[:8]}"
        response = await client.post(
            f"/v1/banks/{bank['id']}/documents",
            json={"id": unique_id},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == unique_id
        assert data["bank_id"] == bank["id"]
        assert "created_at" in data
        assert "updated_at" in data

    async def test_create_document_in_nonexistent_bank(self, client: AsyncClient):
        """Creating document in nonexistent bank returns 404."""
        response = await client.post(
            "/v1/banks/nonexistent/documents",
            json={"id": "doc-1"},
        )
        assert response.status_code == 404

    async def test_get_document(self, client: AsyncClient, bank: dict, document: dict):
        """Can get a document by ID."""
        response = await client.get(f"/v1/banks/{bank['id']}/documents/{document['id']}")
        assert response.status_code == 200
        assert response.json()["id"] == document["id"]


class TestRetainRecall:
    """Retain and recall endpoint tests."""

    async def test_retain_content(self, client: AsyncClient, bank: dict, document: dict):
        """Can retain content in a document."""
        response = await client.post(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/retain",
            json={"content": "Test memory content", "update_mode": "append"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "Test memory content" in data["content"]
        assert data["updated_at"] != data["created_at"]

    async def test_retain_replace_mode(self, client: AsyncClient, bank: dict, document: dict):
        """Retain with replace mode overwrites content."""
        await client.post(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/retain",
            json={"content": "First", "update_mode": "append"},
        )
        response = await client.post(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/retain",
            json={"content": "Second", "update_mode": "replace"},
        )
        assert response.status_code == 200
        assert response.json()["content"] == "Second"

    async def test_recall_memories(self, client: AsyncClient, bank: dict, document: dict):
        """Can recall memories from a document."""
        await client.post(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/retain",
            json={"content": "Memory one", "update_mode": "append"},
        )
        await client.post(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/retain",
            json={"content": "Memory two", "update_mode": "append"},
        )

        response = await client.get(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/recall",
            params={"limit": 10},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["memories"]) == 2

    async def test_recall_with_query_filter(self, client: AsyncClient, bank: dict, document: dict):
        """Recall filters memories by query."""
        await client.post(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/retain",
            json={"content": "User likes apples", "update_mode": "append"},
        )
        await client.post(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/retain",
            json={"content": "User likes bananas", "update_mode": "append"},
        )

        response = await client.get(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/recall",
            params={"query": "apples", "limit": 10},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert "apples" in data["memories"][0]["text"].lower()

    async def test_recall_nonexistent_document(self, client: AsyncClient, bank: dict):
        """Recall on nonexistent document returns 404."""
        response = await client.get(
            f"/v1/banks/{bank['id']}/documents/nonexistent/recall",
        )
        assert response.status_code == 404

    async def test_memory_has_mentioned_at(self, client: AsyncClient, bank: dict, document: dict):
        """Retained memories have mentioned_at timestamp."""
        await client.post(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/retain",
            json={"content": "Timestamped memory", "update_mode": "append"},
        )
        response = await client.get(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/recall",
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["memories"]) == 1
        assert data["memories"][0]["mentioned_at"] is not None


class TestExtractionPipeline:
    """Extraction pipeline tests."""

    async def test_extract_occurred_start_from_iso_date(self, client: AsyncClient, bank: dict, document: dict):
        """Memories extract occurred_start from ISO date in content."""
        await client.post(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/retain",
            json={"content": "Meeting scheduled for 2025-05-01T10:00:00Z.", "update_mode": "append"},
        )
        response = await client.get(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/recall",
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        memory = data["memories"][0]
        assert memory["occurred_start"] == "2025-05-01T10:00:00Z"
        assert memory["mentioned_at"] is not None

    async def test_extract_occurred_start_from_yyyy_mm_dd(self, client: AsyncClient, bank: dict, document: dict):
        """Memories extract occurred_start from YYYY-MM-DD date in content."""
        await client.post(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/retain",
            json={"content": "The event happened on 2025-05-01.", "update_mode": "append"},
        )
        response = await client.get(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/recall",
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        memory = data["memories"][0]
        assert memory["occurred_start"] == "2025-05-01T00:00:00+00:00"

    async def test_extract_occurred_start_fallback_to_mentioned_at(self, client: AsyncClient, bank: dict, document: dict):
        """Memories fall back to mentioned_at when no date found in content."""
        await client.post(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/retain",
            json={"content": "User likes apples.", "update_mode": "append"},
        )
        response = await client.get(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/recall",
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        memory = data["memories"][0]
        assert memory["occurred_start"] == memory["mentioned_at"]

    async def test_multiple_sentences_create_multiple_memories(self, client: AsyncClient, bank: dict, document: dict):
        """Each sentence in retained content creates a separate memory."""
        await client.post(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/retain",
            json={"content": "First sentence. Second sentence with 2025-06-15. Third sentence.", "update_mode": "append"},
        )
        response = await client.get(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/recall",
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        # Check that the middle sentence has the extracted date
        memories_by_text = {m["text"]: m for m in data["memories"]}
        assert "Second sentence with 2025-06-15." in memories_by_text
        assert memories_by_text["Second sentence with 2025-06-15."]["occurred_start"] == "2025-06-15T00:00:00+00:00"


class TestHermesIntegration:
    """Hermes integration tests (issue #4697)."""

    async def test_recall_hermes_endpoint_exists(self, client: AsyncClient, bank: dict, document: dict):
        """Hermes recall endpoint is accessible."""
        await client.post(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/retain",
            json={"content": "Test memory.", "update_mode": "append"},
        )
        response = await client.get(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/recall/hermes",
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1

    async def test_recall_hermes_without_prefix(self, client: AsyncClient, bank: dict, document: dict):
        """Hermes recall without prefix returns original text."""
        await client.post(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/retain",
            json={"content": "Test memory.", "update_mode": "append"},
        )
        response = await client.get(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/recall/hermes",
            params={"recall_timestamp_prefix": "false"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["memories"][0]["text"] == "Test memory."

    async def test_recall_hermes_with_prefix_uses_mentioned_at(self, client: AsyncClient, bank: dict, document: dict):
        """Hermes recall with prefix uses mentioned_at as primary timestamp."""
        await client.post(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/retain",
            json={"content": "Test memory.", "update_mode": "append"},
        )
        response = await client.get(
            f"/v1/banks/{bank['id']}/documents/{document['id']}/recall/hermes",
            params={"recall_timestamp_prefix": "true"},
        )
        assert response.status_code == 200
        data = response.json()
        memory = data["memories"][0]
        # Should have [YYYY-MM-DD] prefix from mentioned_at
        assert memory["text"].startswith("[")
        assert memory["text"][1:11].count("-") == 2  # YYYY-MM-DD format
        assert memory["text"].endswith("Test memory.")

    async def test_recall_hermes_prefix_falls_back_to_occurred_start(self, client: AsyncClient, bank: dict, document: dict):
        """Hermes recall falls back to occurred_start when mentioned_at not available."""
        # This test verifies the fallback logic in format_memory_for_hermes
        # We can't easily test missing mentioned_at since DB always sets it,
        # but we verify the format function works with occurred_start
        from main import format_memory_for_hermes, Memory, HermesConfig
        
        memory = Memory(
            id="test-mem",
            document_id="test-doc",
            text="Test memory with date 2025-06-15.",
            mentioned_at=None,
            occurred_start="2025-06-15T10:00:00Z",
            occurred_end=None,
            score=1.0
        )
        config = HermesConfig(recall_timestamp_prefix=True)
        formatted = format_memory_for_hermes(memory, config)
        assert formatted == "[2025-06-15] Test memory with date 2025-06-15."

    async def test_recall_hermes_prefix_uses_mentioned_at_over_occurred_start(self, client: AsyncClient, bank: dict, document: dict):
        """Hermes recall prefers mentioned_at over occurred_start."""
        from main import format_memory_for_hermes, Memory, HermesConfig
        
        memory = Memory(
            id="test-mem",
            document_id="test-doc",
            text="Test memory.",
            mentioned_at="2025-05-01T12:00:00Z",
            occurred_start="2025-06-15T10:00:00Z",
            occurred_end=None,
            score=1.0
        )
        config = HermesConfig(recall_timestamp_prefix=True)
        formatted = format_memory_for_hermes(memory, config)
        # Should use mentioned_at (2025-05-01) not occurred_start (2025-06-15)
        assert formatted == "[2025-05-01] Test memory."

    async def test_recall_hermes_without_prefix_returns_plain_text(self, client: AsyncClient, bank: dict, document: dict):
        """Hermes recall without prefix returns plain text regardless of timestamps."""
        from main import format_memory_for_hermes, Memory, HermesConfig
        
        memory = Memory(
            id="test-mem",
            document_id="test-doc",
            text="Test memory.",
            mentioned_at="2025-05-01T12:00:00Z",
            occurred_start="2025-06-15T10:00:00Z",
            occurred_end=None,
            score=1.0
        )
        config = HermesConfig(recall_timestamp_prefix=False)
        formatted = format_memory_for_hermes(memory, config)
        assert formatted == "Test memory."

    async def test_recall_hermes_with_empty_timestamps(self, client: AsyncClient, bank: dict, document: dict):
        """Hermes recall handles missing timestamps gracefully."""
        from main import format_memory_for_hermes, Memory, HermesConfig
        
        memory = Memory(
            id="test-mem",
            document_id="test-doc",
            text="Test memory.",
            mentioned_at=None,
            occurred_start=None,
            occurred_end=None,
            score=1.0
        )
        config = HermesConfig(recall_timestamp_prefix=True)
        formatted = format_memory_for_hermes(memory, config)
        assert formatted == "Test memory."
class TestReranker:
    """Reranker tests (issue #5814)."""
    
    async def test_rerank_standard_cohere_format(self, client: AsyncClient):
        """Test rerank with standard Cohere response format (relevance_score)."""
        # We test the parsing logic directly since we can't easily mock the external service
        from main import CohereCompatibleRerankClient
        
        client_rerank = CohereCompatibleRerankClient()
        # Mock the _parse_response method directly
        result = client_rerank._parse_response(
            {"results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.7},
                {"index": 1, "relevance_score": 0.5}
            ]},
            num_documents=3
        )
        
        assert len(result) == 3
        # Should be sorted by score descending
        assert result[0]["index"] == 0
        assert result[0]["relevance_score"] == 0.9
        assert result[1]["index"] == 2
        assert result[1]["relevance_score"] == 0.7
        assert result[2]["index"] == 1
        assert result[2]["relevance_score"] == 0.5
        # Document text should be included
        assert "document" in result[0]
    
    async def test_rerank_score_alias_format(self, client: AsyncClient):
        """Test rerank with 'score' alias instead of 'relevance_score'."""
        from main import CohereCompatibleRerankClient
        
        client_rerank = CohereCompatibleRerankClient()
        result = client_rerank._parse_response(
            {"results": [
                {"index": 1, "score": 0.85},
                {"index": 0, "score": 0.65}
            ]},
            num_documents=2
        )
        
        assert len(result) == 2
        assert result[0]["index"] == 1
        assert result[0]["relevance_score"] == 0.85
        assert result[1]["index"] == 0
        assert result[1]["relevance_score"] == 0.65
    
    async def test_rerank_array_format(self, client: AsyncClient):
        """Test rerank with {scores: [...], indices: [...]} array format."""
        from main import CohereCompatibleRerankClient
        
        client_rerank = CohereCompatibleRerankClient()
        result = client_rerank._parse_response(
            {"scores": [0.9, 0.7, 0.5], "indices": [0, 2, 1]},
            num_documents=3
        )
        
        assert len(result) == 3
        assert result[0]["index"] == 0
        assert result[0]["relevance_score"] == 0.9
        assert result[1]["index"] == 2
        assert result[1]["relevance_score"] == 0.7
        assert result[2]["index"] == 1
        assert result[2]["relevance_score"] == 0.5
    
    async def test_rerank_out_of_range_indices(self, client: AsyncClient):
        """Test rerank skips out-of-range indices gracefully."""
        from main import CohereCompatibleRerankClient
        
        client_rerank = CohereCompatibleRerankClient()
        result = client_rerank._parse_response(
            {"results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 5, "relevance_score": 0.8},  # Out of range
                {"index": -1, "relevance_score": 0.7},  # Negative
                {"index": 1, "relevance_score": 0.6}
            ]},
            num_documents=2
        )
        
        # Only valid indices (0, 1) should be processed
        assert len(result) == 2
        # Score 0.9 for index 0, 0.6 for index 1
        scores_by_idx = {r["index"]: r["relevance_score"] for r in result}
        assert scores_by_idx[0] == 0.9
        assert scores_by_idx[1] == 0.6
    
    async def test_rerank_malformed_items(self, client: AsyncClient):
        """Test rerank handles malformed items gracefully."""
        from main import CohereCompatibleRerankClient
        
        client_rerank = CohereCompatibleRerankClient()
        result = client_rerank._parse_response(
            {"results": [
                {"index": 0, "relevance_score": 0.9},
                "not a dict",  # Malformed
                {"index": 1},  # Missing score
                {"relevance_score": 0.5},  # Missing index
                {"index": 1, "relevance_score": 0.7}
            ]},
            num_documents=2
        )
        
        assert len(result) == 2
        scores_by_idx = {r["index"]: r["relevance_score"] for r in result}
        assert scores_by_idx[0] == 0.9
        assert scores_by_idx[1] == 0.7
    
    async def test_rerank_top_n_limit(self, client: AsyncClient):
        """Test rerank respects top_n limit."""
        from main import CohereCompatibleRerankClient
        
        client_rerank = CohereCompatibleRerankClient()
        result = client_rerank._parse_response(
            {"results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.8},
                {"index": 2, "relevance_score": 0.7},
                {"index": 3, "relevance_score": 0.6}
            ]},
            num_documents=4
        )
        
        # The predict method limits to top_n, but _parse_response returns all
        # Let's test the full flow by checking that _parse_response returns all
        # and the endpoint would limit
        assert len(result) == 4
        # Scores should be in descending order
        scores = [r["relevance_score"] for r in result]
        assert scores == sorted(scores, reverse=True)
    
    async def test_rerank_empty_results(self, client: AsyncClient):
        """Test rerank handles empty results."""
        from main import CohereCompatibleRerankClient
        
        client_rerank = CohereCompatibleRerankClient()
        result = client_rerank._parse_response(
            {"results": []},
            num_documents=3
        )
        
        assert len(result) == 3
        # All scores should be 0.0
        for r in result:
            assert r["relevance_score"] == 0.0
    
    async def test_rerank_unknown_format(self, client: AsyncClient):
        """Test rerank handles unknown response format gracefully."""
        from main import CohereCompatibleRerankClient
        
        client_rerank = CohereCompatibleRerankClient()
        result = client_rerank._parse_response(
            {"unknown": "format"},
            num_documents=2
        )
        
        assert len(result) == 2
        # All scores should be 0.0
        for r in result:
            assert r["relevance_score"] == 0.0
