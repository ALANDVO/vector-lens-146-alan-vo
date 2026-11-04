"""Hermes integration for Hindsight - Memory provider with timestamp prefix support."""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class HermesConfig:
    """Configuration for Hermes memory provider."""
    bank_id: str
    api_base_url: str = "http://localhost:8000"
    recall_limit: int = 10
    recall_tag_prefixes: bool = False
    recall_timestamp_prefix: bool = False


@dataclass
class MemoryResult:
    """A recalled memory from Hindsight."""
    id: str
    text: str
    mentioned_at: Optional[str] = None
    occurred_start: Optional[str] = None
    occurred_end: Optional[str] = None
    occurred_at: Optional[str] = None
    tags: list[str] = field(default_factory=list)


def _fmt_timestamp_prefix(memory: MemoryResult) -> str:
    """
    Format timestamp prefix for a memory.
    Prefers mentioned_at, falls back to occurred_start.
    Returns empty string if no timestamp available.
    """
    ts = memory.mentioned_at or memory.occurred_start
    if ts:
        # Extract date portion YYYY-MM-DD
        date_part = ts[:10]
        return f"[{date_part}] "
    return ""


def format_memory_for_recall(memory: MemoryResult, config: HermesConfig) -> str:
    """
    Format a memory for recall injection.
    Includes tag prefix (if enabled) and timestamp prefix (if enabled).
    """
    parts = []
    
    if config.recall_timestamp_prefix:
        parts.append(_fmt_timestamp_prefix(memory))
    
    parts.append(memory.text)
    
    return "".join(parts)


def format_auto_recall_block(memories: list[MemoryResult], config: HermesConfig) -> str:
    """
    Format memories for auto-recall block injection.
    Each memory on its own line with bullet prefix.
    """
    if not memories:
        return ""
    
    lines = []
    for memory in memories:
        formatted = format_memory_for_recall(memory, config)
        lines.append(f"- {formatted}")
    
    return "\n".join(lines)


def format_recall_tool_output(memories: list[MemoryResult], config: HermesConfig) -> str:
    """
    Format memories for hindsight_recall tool output.
    Each memory numbered with prefix.
    """
    if not memories:
        return "No memories found."
    
    lines = []
    for i, memory in enumerate(memories, 1):
        formatted = format_memory_for_recall(memory, config)
        lines.append(f"{i}. {formatted}")
    
    return "\n".join(lines)


class HermesMemoryProvider:
    """
    Hermes memory provider for Hindsight.
    Handles recall with optional timestamp prefix injection.
    """
    
    def __init__(self, config: HermesConfig):
        self.config = config
        # In a real implementation, this would be an HTTP client to Hindsight API
        self._memories: dict[str, list[MemoryResult]] = {}
    
    def add_memory(self, bank_id: str, memory: MemoryResult) -> None:
        """Add a memory to the store (for testing)."""
        if bank_id not in self._memories:
            self._memories[bank_id] = []
        self._memories[bank_id].append(memory)
    
    def recall(self, bank_id: str, query: str, limit: Optional[int] = None) -> list[MemoryResult]:
        """
        Recall memories for a query.
        Simple in-memory implementation for testing.
        """
        bank_memories = self._memories.get(bank_id, [])
        limit = limit or self.config.recall_limit
        query_lower = query.lower()
        
        results = [
            m for m in bank_memories
            if query_lower in m.text.lower()
        ]
        
        return results[:limit]
    
    def get_auto_recall_block(self, bank_id: str, query: str, limit: Optional[int] = None) -> str:
        """Get formatted auto-recall block for injection into prompt."""
        memories = self.recall(bank_id, query, limit)
        return format_auto_recall_block(memories, self.config)
    
    def get_recall_tool_output(self, bank_id: str, query: str, limit: Optional[int] = None) -> str:
        """Get formatted output for hindsight_recall tool."""
        memories = self.recall(bank_id, query, limit)
        return format_recall_tool_output(memories, self.config)


# Example usage and configuration
def create_hermes_provider(
    bank_id: str,
    api_base_url: str = "http://localhost:8000",
    recall_limit: int = 10,
    recall_tag_prefixes: bool = False,
    recall_timestamp_prefix: bool = False,
) -> HermesMemoryProvider:
    """Factory function to create a HermesMemoryProvider with config."""
    config = HermesConfig(
        bank_id=bank_id,
        api_base_url=api_base_url,
        recall_limit=recall_limit,
        recall_tag_prefixes=recall_tag_prefixes,
        recall_timestamp_prefix=recall_timestamp_prefix,
    )
    return HermesMemoryProvider(config)

