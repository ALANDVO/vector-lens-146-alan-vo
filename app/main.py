"""Hindsight API - Agent Memory That Learns"""

from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn


# In-memory storage for the skeleton
class MemoryStore:
    def __init__(self):
        self.banks: dict[str, dict[str, dict]] = {}  # bank_id -> doc_id -> document
        self.memories: dict[str, list[dict]] = {}  # bank_id -> list of memories

    def create_bank(self, bank_id: str) -> None:
        if bank_id not in self.banks:
            self.banks[bank_id] = {}
            self.memories[bank_id] = []

    def create_document(self, bank_id: str, doc_id: str, content: str, metadata: dict = None) -> dict:
        self.create_bank(bank_id)
        doc = {
            "id": doc_id,
            "content": content,
            "metadata": metadata or {},
            "bank_id": bank_id,
        }
        self.banks[bank_id][doc_id] = doc
        return doc

    def get_document(self, bank_id: str, doc_id: str) -> Optional[dict]:
        return self.banks.get(bank_id, {}).get(doc_id)

    def add_memory(self, bank_id: str, memory: dict) -> None:
        self.create_bank(bank_id)
        self.memories[bank_id].append(memory)

    def recall(self, bank_id: str, query: str, limit: int = 10) -> list[dict]:
        # Simple in-memory recall: return memories containing query terms
        bank_memories = self.memories.get(bank_id, [])
        query_lower = query.lower()
        results = [
            m for m in bank_memories
            if query_lower in m.get("text", "").lower()
        ]
        return results[:limit]


store = MemoryStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown


app = FastAPI(
    title="Hindsight API",
    description="Agent Memory That Learns",
    version="0.1.0",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class DocumentCreate(BaseModel):
    id: str = Field(..., min_length=1)
    content: str
    metadata: dict = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    id: str
    content: str
    metadata: dict
    bank_id: str


class RecallRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=100)


class MemoryItem(BaseModel):
    id: str
    text: str
    mentioned_at: Optional[str] = None
    occurred_start: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class RecallResponse(BaseModel):
    results: list[MemoryItem]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@app.post(
    "/v1/banks/{bank_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_document(bank_id: str, doc: DocumentCreate):
    """Create a document in a bank."""
    created = store.create_document(bank_id, doc.id, doc.content, doc.metadata)
    return DocumentResponse(**created)


@app.get(
    "/v1/banks/{bank_id}/documents/{doc_id}",
    response_model=DocumentResponse,
)
async def get_document(bank_id: str, doc_id: str):
    """Retrieve a document from a bank."""
    doc = store.get_document(bank_id, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse(**doc)


@app.post(
    "/v1/banks/{bank_id}/recall",
    response_model=RecallResponse,
)
async def recall_memories(bank_id: str, request: RecallRequest):
    """Recall memories for a query."""
    store.create_bank(bank_id)
    results = store.recall(bank_id, request.query, request.limit)
    return RecallResponse(results=[MemoryItem(**r) for r in results])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)