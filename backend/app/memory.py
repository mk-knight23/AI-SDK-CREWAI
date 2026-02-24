"""Memory management for CrewAI agents and tasks."""
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import json
import os


class MemoryEntry(BaseModel):
    """A single memory entry."""

    content: str = Field(description="The content of the memory")
    timestamp: datetime = Field(default_factory=datetime.now)
    source_agent: str = Field(description="Which agent created this memory")
    task_id: Optional[str] = Field(None, description="Associated task ID")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(default=1.0, ge=0.0, le=1.0)


class CrewMemory:
    """Memory system for CrewAI crews."""

    def __init__(self, persist_path: Optional[str] = None):
        """Initialize the memory system.

        Args:
            persist_path: Optional path to persist memory to disk
        """
        self.memories: List[MemoryEntry] = []
        self.persist_path = persist_path
        self.task_memory: Dict[str, List[MemoryEntry]] = {}

        if persist_path and os.path.exists(persist_path):
            self._load_from_disk()

    def add_memory(
        self,
        content: str,
        source_agent: str,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0
    ) -> MemoryEntry:
        """Add a new memory entry.

        Args:
            content: The content of the memory
            source_agent: Which agent created this memory
            task_id: Optional associated task ID
            metadata: Optional metadata dictionary
            importance: Importance score (0.0 to 1.0)

        Returns:
            The created MemoryEntry
        """
        entry = MemoryEntry(
            content=content,
            source_agent=source_agent,
            task_id=task_id,
            metadata=metadata or {},
            importance=importance
        )

        self.memories.append(entry)

        if task_id:
            if task_id not in self.task_memory:
                self.task_memory[task_id] = []
            self.task_memory[task_id].append(entry)

        if self.persist_path:
            self._save_to_disk()

        return entry

    def get_memories(
        self,
        agent: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: Optional[int] = None,
        min_importance: float = 0.0
    ) -> List[MemoryEntry]:
        """Retrieve memories with optional filters.

        Args:
            agent: Filter by source agent
            task_id: Filter by task ID
            limit: Maximum number of memories to return
            min_importance: Minimum importance threshold

        Returns:
            List of matching MemoryEntry objects
        """
        memories = self.memories

        if task_id:
            memories = self.task_memory.get(task_id, memories)
        else:
            memories = [m for m in memories if m.task_id == task_id or task_id is None]

        if agent:
            memories = [m for m in memories if m.source_agent == agent]

        memories = [m for m in memories if m.importance >= min_importance]

        # Sort by timestamp and importance
        memories = sorted(
            memories,
            key=lambda m: (m.importance, m.timestamp),
            reverse=True
        )

        if limit:
            memories = memories[:limit]

        return memories

    def search_memories(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Search memories by content.

        Args:
            query: Search query string
            limit: Maximum results to return

        Returns:
            List of matching MemoryEntry objects
        """
        query_lower = query.lower()
        matching = [
            m for m in self.memories
            if query_lower in m.content.lower()
        ]

        # Sort by relevance (simple keyword matching count)
        matching = sorted(
            matching,
            key=lambda m: (m.content.lower().count(query_lower), m.importance),
            reverse=True
        )

        return matching[:limit]

    def get_context_for_task(self, task_id: str) -> str:
        """Get formatted context string for a task.

        Args:
            task_id: The task ID to get context for

        Returns:
            Formatted context string
        """
        memories = self.get_memories(task_id=task_id)

        if not memories:
            return "No previous context available."

        context_parts = []
        for memory in memories:
            context_parts.append(
                f"[{memory.source_agent}] {memory.content}"
            )

        return "\n".join(context_parts)

    def clear_memories(
        self,
        agent: Optional[str] = None,
        task_id: Optional[str] = None
    ):
        """Clear memories with optional filters.

        Args:
            agent: Filter by agent to clear
            task_id: Filter by task ID to clear
        """
        if task_id:
            if task_id in self.task_memory:
                # Remove from main list
                task_memories = self.task_memory[task_id]
                self.memories = [m for m in self.memories if m not in task_memories]
                del self.task_memory[task_id]
        elif agent:
            self.memories = [m for m in self.memories if m.source_agent != agent]
            # Also clear from task memory
            for task_id in self.task_memory:
                self.task_memory[task_id] = [
                    m for m in self.task_memory[task_id]
                    if m.source_agent != agent
                ]
        else:
            self.memories.clear()
            self.task_memory.clear()

        if self.persist_path:
            self._save_to_disk()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories.

        Returns:
            Dictionary with memory statistics
        """
        agent_counts = {}
        for memory in self.memories:
            agent_counts[memory.source_agent] = agent_counts.get(memory.source_agent, 0) + 1

        return {
            "total_memories": len(self.memories),
            "tasks_with_memory": len(self.task_memory),
            "agent_counts": agent_counts,
            "avg_importance": sum(m.importance for m in self.memories) / len(self.memories) if self.memories else 0
        }

    def _save_to_disk(self):
        """Save memories to disk."""
        if not self.persist_path:
            return

        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)

        data = {
            "memories": [m.model_dump() for m in self.memories],
            "task_memory": {
                task_id: [m.model_dump() for m in memories]
                for task_id, memories in self.task_memory.items()
            }
        }

        with open(self.persist_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _load_from_disk(self):
        """Load memories from disk."""
        if not self.persist_path or not os.path.exists(self.persist_path):
            return

        with open(self.persist_path, 'r') as f:
            data = json.load(f)

        self.memories = [MemoryEntry(**m) for m in data.get("memories", [])]

        self.task_memory = {}
        for task_id, memories in data.get("task_memory", {}).items():
            self.task_memory[task_id] = [MemoryEntry(**m) for m in memories]


class TaskExecutionContext:
    """Context manager for task execution with memory."""

    def __init__(self, memory: CrewMemory, task_id: str, agent_name: str):
        """Initialize the task execution context.

        Args:
            memory: The CrewMemory instance
            task_id: The task ID
            agent_name: The agent executing the task
        """
        self.memory = memory
        self.task_id = task_id
        self.agent_name = agent_name
        self._start_time = None

    def __enter__(self):
        """Enter the context, recording task start."""
        self._start_time = datetime.now()
        self.memory.add_memory(
            content=f"Started task execution",
            source_agent=self.agent_name,
            task_id=self.task_id,
            importance=0.5
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, recording task completion."""
        duration = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0

        if exc_type is None:
            self.memory.add_memory(
                content=f"Completed task execution (duration: {duration:.2f}s)",
                source_agent=self.agent_name,
                task_id=self.task_id,
                importance=0.5
            )
        else:
            self.memory.add_memory(
                content=f"Task execution failed: {exc_val}",
                source_agent=self.agent_name,
                task_id=self.task_id,
                importance=0.8,
                metadata={"error": str(exc_val)}
            )

        return False  # Don't suppress exceptions


# Global memory instance
crew_memory = CrewMemory(persist_path="./data/crew_memory.json")
