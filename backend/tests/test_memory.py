"""Tests for CrewAI memory management system."""
import pytest
from unittest.mock import patch, MagicMock
from app.memory import MemoryEntry, CrewMemory, TaskExecutionContext
from datetime import datetime
import json
import tempfile


class TestMemoryEntry:
    """Test suite for MemoryEntry model."""

    def test_create_memory_entry(self):
        """Test creating a basic memory entry."""
        entry = MemoryEntry(
            content="Test memory content",
            source_agent="researcher"
        )

        assert entry.content == "Test memory content"
        assert entry.source_agent == "researcher"
        assert entry.importance == 1.0
        assert entry.task_id is None
        assert entry.metadata == {}

    def test_memory_entry_with_all_fields(self):
        """Test creating memory entry with all fields."""
        now = datetime.now()
        entry = MemoryEntry(
            content="Full memory entry",
            source_agent="writer",
            task_id="task-123",
            metadata={"key": "value"},
            importance=0.8
        )

        assert entry.content == "Full memory entry"
        assert entry.source_agent == "writer"
        assert entry.task_id == "task-123"
        assert entry.metadata == {"key": "value"}
        assert entry.importance == 0.8

    def test_importance_validation(self):
        """Test importance score validation."""
        # Valid ranges
        MemoryEntry(content="Test", source_agent="agent", importance=0.0)
        MemoryEntry(content="Test", source_agent="agent", importance=0.5)
        MemoryEntry(content="Test", source_agent="agent", importance=1.0)

        # Invalid ranges
        with pytest.raises(ValueError):
            MemoryEntry(content="Test", source_agent="agent", importance=-0.1)

        with pytest.raises(ValueError):
            MemoryEntry(content="Test", source_agent="agent", importance=1.1)


class TestCrewMemory:
    """Test suite for CrewMemory class."""

    @pytest.fixture
    def memory(self):
        """Create a fresh CrewMemory instance for each test."""
        return CrewMemory()

    def test_init(self, memory):
        """Test memory initialization."""
        assert memory.memories == []
        assert memory.task_memory == {}

    def test_add_basic_memory(self, memory):
        """Test adding a basic memory."""
        entry = memory.add_memory(
            content="Test content",
            source_agent="researcher"
        )

        assert entry in memory.memories
        assert entry.content == "Test content"
        assert entry.source_agent == "researcher"

    def test_add_memory_with_task_id(self, memory):
        """Test adding memory with task ID."""
        entry = memory.add_memory(
            content="Task memory",
            source_agent="writer",
            task_id="task-1"
        )

        assert entry in memory.memories
        assert "task-1" in memory.task_memory
        assert entry in memory.task_memory["task-1"]

    def test_add_memory_with_metadata(self, memory):
        """Test adding memory with metadata."""
        entry = memory.add_memory(
            content="Metadata test",
            source_agent="analyst",
            metadata={"count": 5, "tags": ["test", "data"]}
        )

        assert entry.metadata == {"count": 5, "tags": ["test", "data"]}

    def test_get_all_memories(self, memory):
        """Test retrieving all memories."""
        memory.add_memory("Content 1", "agent1")
        memory.add_memory("Content 2", "agent2")
        memory.add_memory("Content 3", "agent1")

        memories = memory.get_memories()

        assert len(memories) == 3

    def test_get_memories_by_agent(self, memory):
        """Test filtering memories by agent."""
        memory.add_memory("Content 1", "researcher")
        memory.add_memory("Content 2", "writer")
        memory.add_memory("Content 3", "researcher")

        researcher_memories = memory.get_memories(agent="researcher")
        writer_memories = memory.get_memories(agent="writer")

        assert len(researcher_memories) == 2
        assert len(writer_memories) == 1

    def test_get_memories_by_task_id(self, memory):
        """Test filtering memories by task ID."""
        memory.add_memory("Task 1 content", "agent1", task_id="task-1")
        memory.add_memory("Task 2 content", "agent2", task_id="task-2")
        memory.add_memory("More task 1", "agent1", task_id="task-1")

        task1_memories = memory.get_memories(task_id="task-1")
        task2_memories = memory.get_memories(task_id="task-2")

        assert len(task1_memories) == 2
        assert len(task2_memories) == 1

    def test_get_memories_with_limit(self, memory):
        """Test limiting number of returned memories."""
        for i in range(10):
            memory.add_memory(f"Content {i}", "agent1")

        memories = memory.get_memories(limit=5)

        assert len(memories) == 5

    def test_get_memories_with_importance_filter(self, memory):
        """Test filtering by minimum importance."""
        memory.add_memory("Low importance", "agent1", importance=0.3)
        memory.add_memory("High importance", "agent2", importance=0.9)
        memory.add_memory("Medium importance", "agent3", importance=0.6)

        high_importance = memory.get_memories(min_importance=0.7)

        assert len(high_importance) == 1
        assert high_importance[0].content == "High importance"

    def test_search_memories(self, memory):
        """Test searching memories by content."""
        memory.add_memory("Python is great for data science", "agent1")
        memory.add_memory("JavaScript is used for web development", "agent2")
        memory.add_memory("Python has many libraries", "agent3")

        python_results = memory.search_memories("Python")

        assert len(python_results) == 2

    def test_search_memories_with_limit(self, memory):
        """Test search with result limit."""
        for i in range(10):
            memory.add_memory(f"Test content about Python {i}", "agent1")

        results = memory.search_memories("Python", limit=3)

        assert len(results) == 3

    def test_get_context_for_task(self, memory):
        """Test getting formatted context for a task."""
        memory.add_memory("Research finding 1", "researcher", task_id="task-1")
        memory.add_memory("Draft content", "writer", task_id="task-1")
        memory.add_memory("Review comments", "reviewer", task_id="task-1")

        context = memory.get_context_for_task("task-1")

        assert "[researcher] Research finding 1" in context
        assert "[writer] Draft content" in context
        assert "[reviewer] Review comments" in context

    def test_get_context_for_empty_task(self, memory):
        """Test getting context for task with no memories."""
        context = memory.get_context_for_task("nonexistent-task")

        assert context == "No previous context available."

    def test_clear_all_memories(self, memory):
        """Test clearing all memories."""
        memory.add_memory("Content 1", "agent1", task_id="task-1")
        memory.add_memory("Content 2", "agent2", task_id="task-2")

        memory.clear_memories()

        assert len(memory.memories) == 0
        assert len(memory.task_memory) == 0

    def test_clear_memories_by_agent(self, memory):
        """Test clearing memories for specific agent."""
        memory.add_memory("Content 1", "agent1")
        memory.add_memory("Content 2", "agent2")
        memory.add_memory("Content 3", "agent1")

        memory.clear_memories(agent="agent1")

        assert len(memory.memories) == 1
        assert memory.memories[0].source_agent == "agent2"

    def test_clear_memories_by_task(self, memory):
        """Test clearing memories for specific task."""
        memory.add_memory("Task 1 content", "agent1", task_id="task-1")
        memory.add_memory("Task 2 content", "agent2", task_id="task-2")

        memory.clear_memories(task_id="task-1")

        assert "task-1" not in memory.task_memory
        assert "task-2" in memory.task_memory

    def test_get_memory_stats(self, memory):
        """Test getting memory statistics."""
        memory.add_memory("Content 1", "researcher")
        memory.add_memory("Content 2", "writer")
        memory.add_memory("Content 3", "researcher")
        memory.add_memory("Content 4", "analyst")

        stats = memory.get_memory_stats()

        assert stats["total_memories"] == 4
        assert stats["tasks_with_memory"] == 0
        assert stats["agent_counts"]["researcher"] == 2
        assert stats["agent_counts"]["writer"] == 1
        assert stats["agent_counts"]["analyst"] == 1

    def test_get_memory_stats_with_tasks(self, memory):
        """Test memory stats with task memories."""
        memory.add_memory("Task 1", "agent1", task_id="task-1")
        memory.add_memory("Task 2", "agent2", task_id="task-2")

        stats = memory.get_memory_stats()

        assert stats["tasks_with_memory"] == 2

    def test_persist_to_disk(self):
        """Test saving memories to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = f"{tmpdir}/memory.json"
            memory = CrewMemory(persist_path=persist_path)

            memory.add_memory("Test content", "researcher", task_id="task-1")

            # Verify file was created
            import os
            assert os.path.exists(persist_path)

            # Verify content
            with open(persist_path, 'r') as f:
                data = json.load(f)

            assert len(data["memories"]) == 1
            assert data["memories"][0]["content"] == "Test content"

    def test_load_from_disk(self):
        """Test loading memories from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = f"{tmpdir}/memory.json"

            # Create initial memory and save
            memory1 = CrewMemory(persist_path=persist_path)
            memory1.add_memory("Saved content", "writer", importance=0.8)
            memory1._save_to_disk()

            # Create new memory instance and load
            memory2 = CrewMemory(persist_path=persist_path)

            assert len(memory2.memories) == 1
            assert memory2.memories[0].content == "Saved content"
            assert memory2.memories[0].importance == 0.8


class TestTaskExecutionContext:
    """Test suite for TaskExecutionContext class."""

    @pytest.fixture
    def memory(self):
        """Create a CrewMemory instance."""
        return CrewMemory()

    def test_successful_task_context(self, memory):
        """Test context manager with successful execution."""
        with TaskExecutionContext(memory, "task-1", "researcher"):
            pass

        memories = memory.get_memories(task_id="task-1")

        assert len(memories) == 2
        memory_contents = [m.content for m in memories]
        assert any("Started task execution" in c for c in memory_contents)
        assert any("Completed task execution" in c for c in memory_contents)

    def test_failed_task_context(self, memory):
        """Test context manager with failed execution."""
        try:
            with TaskExecutionContext(memory, "task-1", "writer"):
                raise ValueError("Task failed!")
        except ValueError:
            pass

        memories = memory.get_memories(task_id="task-1")

        assert len(memories) == 2
        memory_contents = [m.content for m in memories]
        assert any("Started task execution" in c for c in memory_contents)
        assert any("Task execution failed" in c for c in memory_contents)

    def test_context_records_duration(self, memory):
        """Test that context records task duration."""
        import time

        with TaskExecutionContext(memory, "task-1", "analyst"):
            time.sleep(0.1)

        memories = memory.get_memories(task_id="task-1")

        # Find the completion memory
        completion_memory = next(
            (m for m in memories if "Completed" in m.content or "duration" in m.content),
            None
        )
        assert completion_memory is not None
        assert "duration:" in completion_memory.content or "duration" in completion_memory.content

    def test_context_task_id_association(self, memory):
        """Test that context manager associates memories with task."""
        with TaskExecutionContext(memory, "my-task", "planner"):
            pass

        task_memories = memory.get_memories(task_id="my-task")

        assert len(task_memories) == 2
        assert all(m.task_id == "my-task" for m in task_memories)
