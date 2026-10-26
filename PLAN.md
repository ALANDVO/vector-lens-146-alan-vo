# Product development plan

Each round completes one feature or repair. Future items are planned, not implemented claims.

## Current round

api-skeleton

## Roadmap

### api-skeleton — FastAPI skeleton with health check and memory CRUD

Status: done

Acceptance:
- GET /health returns 200 OK
- POST /v1/banks/{bank_id}/documents creates a document
- GET /v1/banks/{bank_id}/documents/{doc_id} retrieves a document
- POST /v1/banks/{bank_id}/recall returns memories for a query

Implementation: `app/main.py`

Tests: `tests/test_main.py`

### hermes-recall-timestamp — Hermes recall timestamp prefix injection

Status: planned

Acceptance:
- recall_timestamp_prefix setting toggles date prefix on recalled memories
- Auto-recall block shows [YYYY-MM-DD] prefix when enabled
- hindsight_recall tool shows [YYYY-MM-DD] prefix when enabled
- mentioned_at preferred, occurred_start fallback

### reranker-cohere-compat — Cohere-compatible reranker response parsing

Status: planned

Acceptance:
- Handles 'score' alias for 'relevance_score'
- Handles {scores, indices} array response shape
- Skips out-of-range indices instead of crashing
- Tests cover all three failure modes

### livekit-integration — LiveKit Agents integration package

Status: planned

Acceptance:
- HindsightMemory class for session memory
- HindsightAgent subclass with automatic recall
- create_hindsight_tools() for function tools
- Unit tests with mocked client

