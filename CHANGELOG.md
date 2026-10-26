# Changelog

## 1.0.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-26

### Added
- FastAPI skeleton with health check endpoint (`GET /health`)
- Document CRUD endpoints:
  - `POST /v1/banks/{bank_id}/documents` - Create document
  - `GET /v1/banks/{bank_id}/documents/{doc_id}` - Retrieve document
- Memory recall endpoint:
  - `POST /v1/banks/{bank_id}/recall` - Query memories
- In-memory storage for banks, documents, and memories
- Test suite with 5 passing tests covering all endpoints
- Project configuration with pyproject.toml
- README with API documentation and quick start guide

### Infrastructure
- Python 3.11+ requirement
- Dependencies: fastapi, uvicorn, pydantic, pydantic-settings
- Dev dependencies: pytest, pytest-asyncio, httpx
- Git repository initialized