# vector-lens-146-alan-vo

> Work in progress. This checkpoint may be incomplete or fail tests. See PLAN.md and STATUS.md before use.


**Hindsight: Agent Memory That Learns**

An AI agent memory system with persistent storage, recall, and reflection capabilities.

## Features Implemented

- **FastAPI REST API** - Health check, document CRUD, and memory recall endpoints
- **In-memory storage** - Banks, documents, and memories with simple query-based recall

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the server
python -m app.main

# Or with uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/v1/banks/{bank_id}/documents` | Create a document |
| GET | `/v1/banks/{bank_id}/documents/{doc_id}` | Retrieve a document |
| POST | `/v1/banks/{bank_id}/recall` | Recall memories for a query |

## Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Create a document
curl -X POST http://localhost:8000/v1/banks/my-bank/documents \
  -H "Content-Type: application/json" \
  -d '{"id": "doc-1", "content": "User prefers coffee over tea", "metadata": {"source": "chat"}}'

# Retrieve a document
curl http://localhost:8000/v1/banks/my-bank/documents/doc-1

# Recall memories
curl -X POST http://localhost:8000/v1/banks/my-bank/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "coffee", "limit": 10}'
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
vector-lens-146-alan-vo/
├── app/
│   └── main.py          # FastAPI application
├── tests/
│   └── test_main.py     # API tests
├── feature-plan.json    # Feature tracking
├── PLAN.md              # Implementation plan
├── pyproject.toml       # Project configuration
└── README.md            # This file
```

## License

MIT
Maintainer: Alan Vo <alanvo@gmail.com>
