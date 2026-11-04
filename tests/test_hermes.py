"""Tests for Hermes integration with timestamp prefix."""

import pytest
from app.hermes import (
    HermesConfig,
    MemoryResult,
    _fmt_timestamp_prefix,
    format_memory_for_recall,
    format_auto_recall_block,
    format_recall_tool_output,
    HermesMemoryProvider,
)


class TestTimestampPrefix:
    """Tests for _fmt_timestamp_prefix function."""

    def test_fmt_timestamp_prefix_mentioned_at(self):
        """_fmt_timestamp_prefix should prefer mentioned_at."""
        memory = MemoryResult(
            id="mem-1", 
            text="Test", 
            mentioned_at="2024-01-15T10:00:00Z",
            occurred_start="2024-01-10T10:00:00Z"
        )
        assert _fmt_timestamp_prefix(memory) == "[2024-01-15] "

    def test_fmt_timestamp_prefix_occurred_start_fallback(self):
        """_fmt_timestamp_prefix should fall back to occurred_start."""
        memory = MemoryResult(
            id="mem-1", 
            text="Test", 
            occurred_start="2024-01-10T10:00:00Z"
        )
        assert _fmt_timestamp_prefix(memory) == "[2024-01-10] "

    def test_fmt_timestamp_prefix_none(self):
        """_fmt_timestamp_prefix should return empty string when no timestamp."""
        memory = MemoryResult(id="mem-1", text="Test")
        assert _fmt_timestamp_prefix(memory) == ""


class TestFormatMemoryForRecall:
    """Tests for format_memory_for_recall function."""

    def test_without_timestamp_prefix(self):
        """Should return plain text when recall_timestamp_prefix is False."""
        config = HermesConfig(bank_id="test", recall_timestamp_prefix=False)
        memory = MemoryResult(id="mem-1", text="User likes coffee", mentioned_at="2024-01-15T10:00:00Z")
        assert format_memory_for_recall(memory, config) == "User likes coffee"

    def test_with_timestamp_prefix_mentioned_at(self):
        """Should prefix with mentioned_at date when enabled."""
        config = HermesConfig(bank_id="test", recall_timestamp_prefix=True)
        memory = MemoryResult(id="mem-1", text="User likes coffee", mentioned_at="2024-01-15T10:00:00Z")
        assert format_memory_for_recall(memory, config) == "[2024-01-15] User likes coffee"

    def test_with_timestamp_prefix_occurred_start(self):
        """Should prefix with occurred_start date when mentioned_at not available."""
        config = HermesConfig(bank_id="test", recall_timestamp_prefix=True)
        memory = MemoryResult(id="mem-1", text="User prefers tea", occurred_start="2024-01-16T10:00:00Z")
        assert format_memory_for_recall(memory, config) == "[2024-01-16] User prefers tea"


class TestFormatAutoRecallBlock:
    """Tests for format_auto_recall_block function."""

    def test_empty_memories(self):
        """Should return empty string for no memories."""
        config = HermesConfig(bank_id="test", recall_timestamp_prefix=True)
        assert format_auto_recall_block([], config) == ""

    def test_single_memory_with_timestamp(self):
        """Should format single memory with timestamp prefix."""
        config = HermesConfig(bank_id="test", recall_timestamp_prefix=True)
        memories = [MemoryResult(id="mem-1", text="User likes coffee", mentioned_at="2024-01-15T10:00:00Z")]
        assert format_auto_recall_block(memories, config) == "- [2024-01-15] User likes coffee"

    def test_multiple_memories(self):
        """Should format multiple memories each on own line."""
        config = HermesConfig(bank_id="test", recall_timestamp_prefix=True)
        memories = [
            MemoryResult(id="mem-1", text="User likes coffee", mentioned_at="2024-01-15T10:00:00Z"),
            MemoryResult(id="mem-2", text="User prefers tea", mentioned_at="2024-01-16T10:00:00Z"),
        ]
        expected = "- [2024-01-15] User likes coffee\n- [2024-01-16] User prefers tea"
        assert format_auto_recall_block(memories, config) == expected


class TestFormatRecallToolOutput:
    """Tests for format_recall_tool_output function."""

    def test_empty_memories(self):
        """Should return 'No memories found.' for empty list."""
        config = HermesConfig(bank_id="test", recall_timestamp_prefix=True)
        assert format_recall_tool_output([], config) == "No memories found."

    def test_single_memory_with_timestamp(self):
        """Should format single memory with numbered prefix and timestamp."""
        config = HermesConfig(bank_id="test", recall_timestamp_prefix=True)
        memories = [MemoryResult(id="mem-1", text="User likes coffee", mentioned_at="2024-01-15T10:00:00Z")]
        assert format_recall_tool_output(memories, config) == "1. [2024-01-15] User likes coffee"

    def test_multiple_memories_numbered(self):
        """Should format multiple memories with numbered prefixes."""
        config = HermesConfig(bank_id="test", recall_timestamp_prefix=True)
        memories = [
            MemoryResult(id="mem-1", text="User likes coffee", mentioned_at="2024-01-15T10:00:00Z"),
            MemoryResult(id="mem-2", text="User prefers tea", mentioned_at="2024-01-16T10:00:00Z"),
        ]
        expected = "1. [2024-01-15] User likes coffee\n2. [2024-01-16] User prefers tea"
        assert format_recall_tool_output(memories, config) == expected


class TestHermesMemoryProvider:
    """Tests for HermesMemoryProvider class."""

    def test_recall_without_timestamp_prefix(self):
        """Recall should return plain memories when timestamp prefix disabled."""
        provider = HermesMemoryProvider(HermesConfig(bank_id="test-bank", recall_timestamp_prefix=False))
        provider.add_memory("test-bank", MemoryResult(
            id="mem-1", text="User likes coffee", mentioned_at="2024-01-15T10:00:00Z"
        ))
        
        memories = provider.recall("test-bank", "coffee")
        assert len(memories) == 1
        assert memories[0].text == "User likes coffee"

    def test_auto_recall_block_without_timestamp(self):
        """Auto-recall block should not have timestamp when disabled."""
        provider = HermesMemoryProvider(HermesConfig(bank_id="test-bank", recall_timestamp_prefix=False))
        provider.add_memory("test-bank", MemoryResult(
            id="mem-1", text="User likes coffee", mentioned_at="2024-01-15T10:00:00Z"
        ))
        
        block = provider.get_auto_recall_block("test-bank", "coffee")
        assert block == "- User likes coffee"

    def test_auto_recall_block_with_timestamp(self):
        """Auto-recall block should have timestamp prefix when enabled."""
        provider = HermesMemoryProvider(HermesConfig(bank_id="test-bank", recall_timestamp_prefix=True))
        provider.add_memory("test-bank", MemoryResult(
            id="mem-1", text="User prefers tea", mentioned_at="2024-01-16T10:00:00Z"
        ))
        
        block = provider.get_auto_recall_block("test-bank", "tea")
        assert block == "- [2024-01-16] User prefers tea"

    def test_recall_tool_output_with_timestamp(self):
        """Recall tool output should have timestamp prefix when enabled."""
        provider = HermesMemoryProvider(HermesConfig(bank_id="test-bank", recall_timestamp_prefix=True))
        provider.add_memory("test-bank", MemoryResult(
            id="mem-1", text="User prefers tea", mentioned_at="2024-01-16T10:00:00Z"
        ))
        
        output = provider.get_recall_tool_output("test-bank", "tea")
        assert output == "1. [2024-01-16] User prefers tea"

    def test_recall_uses_mentioned_at_over_occurred_start(self):
        """Recall should use mentioned_at when both timestamps present."""
        provider = HermesMemoryProvider(HermesConfig(bank_id="test-bank", recall_timestamp_prefix=True))
        provider.add_memory("test-bank", MemoryResult(
            id="mem-1", 
            text="Test memory", 
            mentioned_at="2024-01-20T10:00:00Z",
            occurred_start="2024-01-10T10:00:00Z"
        ))
        
        block = provider.get_auto_recall_block("test-bank", "test")
        assert block == "- [2024-01-20] Test memory"

    def test_occurred_start_fallback_when_mentioned_at_missing(self):
        """Should fall back to occurred_start when mentioned_at is None."""
        provider = HermesMemoryProvider(HermesConfig(bank_id="test-bank", recall_timestamp_prefix=True))
        provider.add_memory("test-bank", MemoryResult(
            id="mem-1", 
            text="Test memory", 
            occurred_start="2024-01-10T10:00:00Z"
        ))
        
        block = provider.get_auto_recall_block("test-bank", "test")
        assert block == "- [2024-01-10] Test memory"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
