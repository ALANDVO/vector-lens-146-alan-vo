"""Vector Lens 146 - Hindsight: Agent Memory That Learns"""

from contextlib import asynccontextmanager
from typing import Any, Optional
import sqlite3
import os
import json
import re
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field


DB_PATH = os.environ.get("HINDSIGHT_DB_PATH", "hindsight.db")


def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database schema."""
    conn = get_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS banks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                bank_id TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                update_mode TEXT NOT NULL DEFAULT 'append',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (bank_id) REFERENCES banks(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                text TEXT NOT NULL,
                mentioned_at TEXT,
                occurred_start TEXT,
                occurred_end TEXT,
                score REAL NOT NULL DEFAULT 1.0,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        conn.commit()
    finally:
        conn.close()


# Initialize database at module load for test compatibility
init_db()


class Bank(BaseModel):
    """A memory bank containing documents."""
    id: str
    name: str
    created_at: str = Field(default_factory=lambda: now_iso())


class Document(BaseModel):
    """A document within a bank."""
    id: str = Field(default_factory=lambda: generate_id("doc"))
    bank_id: str = ""
    content: str = ""
    update_mode: str = "append"
    created_at: str = Field(default_factory=lambda: now_iso())
    updated_at: str = Field(default_factory=lambda: now_iso())


class RetainRequest(BaseModel):
    """Request to retain content in a document."""
    content: str = Field(..., description="Content to store")
    update_mode: str = Field(default="append", description="append or replace")


class RecallRequest(BaseModel):
    """Request to recall memories from a document."""
    query: str = Field(default="", description="Optional query to filter memories")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")


class Memory(BaseModel):
    """An extracted memory."""
    id: str
    document_id: str
    text: str
    mentioned_at: Optional[str] = None
    occurred_start: Optional[str] = None
    occurred_end: Optional[str] = None
    score: float = 1.0


class RerankRequest(BaseModel):
    """Request to rerank documents."""
    query: str = Field(..., description="Query to rerank against")
    documents: list[str] = Field(..., description="Documents to rerank")
    top_n: int = Field(default=10, description="Number of top results to return")
    model: str = Field(default="rerank-multilingual-v3.0", description="Reranker model")


class RerankResult(BaseModel):
    """A single reranked result."""
    index: int = Field(..., description="Original index of the document")
    relevance_score: float = Field(..., description="Relevance score")
    document: str = Field(..., description="Document text")


class RerankResponse(BaseModel):
    """Response from rerank endpoint."""
    results: list[RerankResult]


class CohereCompatibleRerankClient:
    """
    Cohere-compatible rerank client that handles multiple response formats.
    
    Handles three response shapes:
    1. Standard Cohere: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
    2. Score alias: {"results": [{"index": 0, "score": 0.9}, ...]}
    3. Array format: {"scores": [0.9, 0.8, ...], "indices": [0, 2, ...]}
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
    
    async def predict(self, query: str, documents: list[str], top_n: int = 10, model: str = "rerank-multilingual-v3.0") -> list[dict]:
        """
        Rerank documents against a query.
        
        Returns a list of dicts with keys: index, relevance_score, document
        """
        import httpx
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "model": model,
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/rerank",
                json=payload,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()
        
        return self._parse_response(result, len(documents))
    
    def _parse_response(self, result: dict, num_documents: int) -> list[dict]:
        """
        Parse rerank response handling multiple formats.
        
        Returns list of dicts with index, relevance_score, document
        """
        all_scores = [0.0] * num_documents
        
        # Format 1 & 2: results array with index and relevance_score/score
        if "results" in result and isinstance(result["results"], list):
            for item in result["results"]:
                if not isinstance(item, dict):
                    continue
                original_idx = item.get("index")
                if original_idx is None:
                    continue
                # Handle out-of-range indices gracefully
                if not isinstance(original_idx, int) or original_idx < 0 or original_idx >= num_documents:
                    continue
                # Try relevance_score first, then score alias
                score = item.get("relevance_score")
                if score is None:
                    score = item.get("score")
                if score is None:
                    continue
                try:
                    all_scores[original_idx] = float(score)
                except (TypeError, ValueError):
                    continue
        
        # Format 3: scores and indices arrays
        elif "scores" in result and "indices" in result:
            scores = result.get("scores", [])
            indices = result.get("indices", [])
            if isinstance(scores, list) and isinstance(indices, list) and len(scores) == len(indices):
                for score, idx in zip(scores, indices):
                    if not isinstance(idx, int) or idx < 0 or idx >= num_documents:
                        continue
                    try:
                        all_scores[idx] = float(score)
                    except (TypeError, ValueError):
                        continue
        
        # Build results with original document text
        results = []
        for idx, score in enumerate(all_scores):
            results.append({
                "index": idx,
                "relevance_score": score,
                "document": documents[idx] if idx < len(documents) else ""
            })
        
        # Sort by score descending and take top_n
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:min(len(results), num_documents)]

class HermesConfig(BaseModel):
    """Configuration for Hermes memory formatting."""
    recall_timestamp_prefix: bool = Field(default=False, description="Prefix memories with timestamp")


def format_memory_for_hermes(memory: Memory, config: HermesConfig) -> str:
    """
    Format a memory for Hermes recall injection.
    If recall_timestamp_prefix is enabled, prefixes with [YYYY-MM-DD] using
    mentioned_at first, then occurred_start as fallback.
    """
    if not config.recall_timestamp_prefix:
        return memory.text
    
    # Get timestamp: mentioned_at first, occurred_start as fallback
    timestamp = memory.mentioned_at or memory.occurred_start
    if not timestamp:
        return memory.text
    
    # Extract date portion (YYYY-MM-DD)
    date_part = timestamp[:10] if len(timestamp) >= 10 else timestamp
    return f"[{date_part}] {memory.text}"


class RecallResponse(BaseModel):
    """Response from recall endpoint."""
    memories: list[Memory]
    total: int


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_id(prefix: str) -> str:
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def extract_occurred_start(text: str, mentioned_at: str) -> Optional[str]:
    """
    Extract occurred_start timestamp from text content.
    Looks for explicit timestamps, dates, or relative time references.
    Falls back to mentioned_at if no explicit time found.
    """
    # Try to find ISO timestamp in text
    iso_pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?'
    match = re.search(iso_pattern, text)
    if match:
        return match.group(0)

    # Try to find date patterns like "2025-05-01" or "May 1, 2025"
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}',  # Month DD, YYYY
        r'\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}',  # DD Month YYYY
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                # Parse and convert to ISO format
                parsed = None
                for fmt in ["%Y-%m-%d", "%b %d, %Y", "%b %d %Y", "%d %b %Y", "%d %B %Y"]:
                    try:
                        parsed = datetime.strptime(match.group(0), fmt).replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        continue
                if parsed:
                    return parsed.isoformat()
            except Exception:
                pass

    # Look for relative time references like "yesterday", "tomorrow", "last week"
    # For now, return mentioned_at as fallback
    return mentioned_at


async def extract_memories(content: str, mentioned_at: str) -> list[dict]:
    """
    Extract structured memories from content using LLM.
    Returns list of memory dicts with text, mentioned_at, occurred_start, occurred_end.
    This is a mockable function for testing - in production would call an LLM.
    """
    # Simple extraction: split by sentences and create a memory for each
    # In production, this would call an LLM with a structured prompt
    memories = []
    
    # Split content into sentences (simple heuristic)
    sentences = re.split(r'(?<=[.!?])\s+', content.strip())
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        occurred_start = extract_occurred_start(sentence, mentioned_at)
        
        memories.append({
            "text": sentence,
            "mentioned_at": mentioned_at,
            "occurred_start": occurred_start,
            "occurred_end": None,
            "score": 1.0
        })
    
    return memories


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - database already initialized at module load
    yield
    # Shutdown


app = FastAPI(
    title="Vector Lens 146 - Hindsight API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/v1/banks", response_model=Bank, status_code=201)
async def create_bank(bank: Bank) -> Bank:
    """Create a new memory bank."""
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO banks (id, name, created_at) VALUES (?, ?, ?)",
            (bank.id, bank.name, bank.created_at)
        )
        conn.commit()
        return bank
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Bank already exists")
    finally:
        conn.close()


@app.get("/v1/banks/{bank_id}", response_model=Bank)
async def get_bank(bank_id: str = Path(..., description="Bank ID")) -> Bank:
    """Get a bank by ID."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM banks WHERE id = ?", (bank_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Bank not found")
        return Bank(id=row["id"], name=row["name"], created_at=row["created_at"])
    finally:
        conn.close()


@app.post("/v1/banks/{bank_id}/documents", response_model=Document, status_code=201)
async def create_document(
    bank_id: str = Path(..., description="Bank ID"),
    document: Document = None,
) -> Document:
    """Create a new document in a bank."""
    conn = get_db()
    try:
        # Verify bank exists
        bank_row = conn.execute(
            "SELECT id FROM banks WHERE id = ?", (bank_id,)
        ).fetchone()
        if not bank_row:
            raise HTTPException(status_code=404, detail="Bank not found")

        # Use provided document or create default
        if document is None:
            document = Document()
        
        doc_id = document.id or generate_id("doc")
        created_at = document.created_at or now_iso()
        updated_at = document.updated_at or now_iso()
        content = document.content or ""
        update_mode = document.update_mode or "append"

        conn.execute(
            """INSERT INTO documents (id, bank_id, content, update_mode, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (doc_id, bank_id, content, update_mode, created_at, updated_at)
        )
        conn.commit()

        return Document(
            id=doc_id,
            bank_id=bank_id,
            content=content,
            update_mode=update_mode,
            created_at=created_at,
            updated_at=updated_at
        )
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Document already exists")
    finally:
        conn.close()


@app.get("/v1/banks/{bank_id}/documents/{doc_id}", response_model=Document)
async def get_document(
    bank_id: str = Path(..., description="Bank ID"),
    doc_id: str = Path(..., description="Document ID"),
) -> Document:
    """Get a document by ID."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ? AND bank_id = ?", (doc_id, bank_id)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found in this bank")
        return Document(
            id=row["id"],
            bank_id=row["bank_id"],
            content=row["content"],
            update_mode=row["update_mode"],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )
    finally:
        conn.close()


@app.post("/v1/banks/{bank_id}/documents/{doc_id}/retain", response_model=Document)
async def retain_content(
    bank_id: str = Path(..., description="Bank ID"),
    doc_id: str = Path(..., description="Document ID"),
    request: RetainRequest = None,
) -> Document:
    """Retain content in a document and extract memories."""
    conn = get_db()
    try:
        # Verify document exists and belongs to bank
        doc_row = conn.execute(
            "SELECT * FROM documents WHERE id = ? AND bank_id = ?", (doc_id, bank_id)
        ).fetchone()
        if not doc_row:
            raise HTTPException(status_code=404, detail="Document not found in this bank")

        content = request.content if request else ""
        update_mode = request.update_mode if request else "append"
        now = now_iso()

        if update_mode == "replace":
            new_content = content
        else:
            # append mode
            old_content = doc_row["content"]
            new_content = old_content + ("\n" if old_content else "") + content

        conn.execute(
            "UPDATE documents SET content = ?, updated_at = ? WHERE id = ?",
            (new_content, now, doc_id)
        )

        # Extract and store memories from new content
        if content:
            memories = await extract_memories(content, now)
            for memory in memories:
                memory_id = generate_id("mem")
                conn.execute(
                    """INSERT INTO memories (id, document_id, text, mentioned_at, occurred_start, occurred_end, score)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (memory_id, doc_id, memory["text"], memory["mentioned_at"], 
                     memory["occurred_start"], memory["occurred_end"], memory["score"])
                )

        conn.commit()

        return Document(
            id=doc_id,
            bank_id=bank_id,
            content=new_content,
            update_mode=update_mode,
            created_at=doc_row["created_at"],
            updated_at=now
        )
    finally:
        conn.close()


@app.get("/v1/banks/{bank_id}/documents/{doc_id}/recall", response_model=RecallResponse)
async def recall_memories(
    bank_id: str = Path(..., description="Bank ID"),
    doc_id: str = Path(..., description="Document ID"),
    query: str = "",
    limit: int = 10,
) -> RecallResponse:
    """Recall memories from a document."""
    conn = get_db()
    try:
        # Verify document exists and belongs to bank
        doc_row = conn.execute(
            "SELECT id FROM documents WHERE id = ? AND bank_id = ?", (doc_id, bank_id)
        ).fetchone()
        if not doc_row:
            raise HTTPException(status_code=404, detail="Document not found in this bank")

        # Count total memories
        if query:
            count_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM memories WHERE document_id = ? AND text LIKE ?",
                (doc_id, f"%{query}%")
            ).fetchone()
            rows = conn.execute(
                """SELECT * FROM memories 
                   WHERE document_id = ? AND text LIKE ?
                   ORDER BY mentioned_at DESC LIMIT ?""",
                (doc_id, f"%{query}%", limit)
            ).fetchall()
        else:
            count_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM memories WHERE document_id = ?",
                (doc_id,)
            ).fetchone()
            rows = conn.execute(
                "SELECT * FROM memories WHERE document_id = ? ORDER BY mentioned_at DESC LIMIT ?",
                (doc_id, limit)
            ).fetchall()

        total = count_row["cnt"] if count_row else 0

        memories_list = [
            Memory(
                id=row["id"],
                document_id=row["document_id"],
                text=row["text"],
                mentioned_at=row["mentioned_at"],
                occurred_start=row["occurred_start"],
                occurred_end=row["occurred_end"],
                score=row["score"]
            )
            for row in rows
        ]

        return RecallResponse(memories=memories_list, total=total)
    finally:
        conn.close()


@app.get("/v1/banks/{bank_id}/documents/{doc_id}/recall/hermes", response_model=RecallResponse)
async def recall_memories_hermes(
    bank_id: str = Path(..., description="Bank ID"),
    doc_id: str = Path(..., description="Document ID"),
    query: str = "",
    limit: int = 10,
    recall_timestamp_prefix: bool = False,
) -> RecallResponse:
    """Recall memories from a document with Hermes formatting (timestamp prefix)."""
    conn = get_db()
    try:
        # Verify document exists and belongs to bank
        doc_row = conn.execute(
            "SELECT id FROM documents WHERE id = ? AND bank_id = ?", (doc_id, bank_id)
        ).fetchone()
        if not doc_row:
            raise HTTPException(status_code=404, detail="Document not found in this bank")

        # Count total memories
        if query:
            count_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM memories WHERE document_id = ? AND text LIKE ?",
                (doc_id, f"%{query}%")
            ).fetchone()
            rows = conn.execute(
                """SELECT * FROM memories 
                   WHERE document_id = ? AND text LIKE ?
                   ORDER BY mentioned_at DESC LIMIT ?""",
                (doc_id, f"%{query}%", limit)
            ).fetchall()
        else:
            count_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM memories WHERE document_id = ?",
                (doc_id,)
            ).fetchone()
            rows = conn.execute(
                "SELECT * FROM memories WHERE document_id = ? ORDER BY mentioned_at DESC LIMIT ?",
                (doc_id, limit)
            ).fetchall()

        total = count_row["cnt"] if count_row else 0

        hermes_config = HermesConfig(recall_timestamp_prefix=recall_timestamp_prefix)

        memories_list = [
            Memory(
                id=row["id"],
                document_id=row["document_id"],
                text=format_memory_for_hermes(
                    Memory(
                        id=row["id"],
                        document_id=row["document_id"],
                        text=row["text"],
                        mentioned_at=row["mentioned_at"],
                        occurred_start=row["occurred_start"],
                        occurred_end=row["occurred_end"],
                        score=row["score"]
                    ),
                    hermes_config
                ),
                mentioned_at=row["mentioned_at"],
                occurred_start=row["occurred_start"],
                occurred_end=row["occurred_end"],
                score=row["score"]
            )
            for row in rows
        ]

        return RecallResponse(memories=memories_list, total=total)
    finally:
        conn.close()


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest) -> RerankResponse:
    """Rerank documents against a query using Cohere-compatible reranker."""
    client = CohereCompatibleRerankClient()
    results = await client.predict(
        query=request.query,
        documents=request.documents,
        top_n=request.top_n,
        model=request.model
    )
    
    rerank_results = [
        RerankResult(
            index=r["index"],
            relevance_score=r["relevance_score"],
            document=r["document"]
        )
        for r in results
    ]
    
    return RerankResponse(results=rerank_results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
