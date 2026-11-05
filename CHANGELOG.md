# Changelog

## 1.2.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

## 1.0.0 — work-in-progress checkpoint

### Added
- **extraction-pipeline**: LLM-based memory extraction from retained content
  - Retain endpoint triggers extraction of structured memories
  - Memories include `mentioned_at` (retention time) and `occurred_start` (extracted from content)
  - Date extraction supports ISO timestamps (2025-05-01T10:00:00Z), YYYY-MM-DD, and month-name formats
  - Falls back to `mentioned_at` when no explicit date found in content
  - Multiple sentences in content create separate memories
  - 4 new extraction tests added

### Fixed
- Document model fields now have proper defaults for test compatibility

## 1.0.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

## 1.0.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.
