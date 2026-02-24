"""Tests for the FastAPI main application with CrewAI endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock


@pytest.fixture
def client():
    """Create a test client for the FastAPI app with mocked dependencies."""
    # Mock the crews and memory modules
    with patch.dict('sys.modules', {
        'app.crews': MagicMock(),
        'app.memory': MagicMock()
    }):
        # Setup mock crew system
        mock_crew_system = MagicMock()
        mock_crew_system.list_agents.return_value = ["researcher", "writer", "reviewer", "analyst", "planner"]
        mock_crew_system.get_agent_info.return_value = {
            "role": "Research Specialist",
            "goal": "Conduct thorough research",
            "backstory": "Expert researcher",
            "allow_delegation": True,
            "verbose": True
        }
        mock_crew_system.execute_sequential.return_value = {
            "status": "completed",
            "result": "Task executed successfully",
            "tasks_count": 1,
            "process": "sequential"
        }
        mock_crew_system.execute_hierarchical.return_value = {
            "status": "completed",
            "result": "Hierarchical execution complete",
            "tasks_count": 2,
            "process": "hierarchical",
            "manager": "planner"
        }
        mock_crew_system.execute_content_pipeline.return_value = {
            "status": "completed",
            "result": "Content pipeline complete",
            "topic": "Test Topic",
            "workflow": "content_pipeline"
        }
        mock_crew_system.create_content_pipeline_crew.return_value = MagicMock()
        mock_crew_system.create_analysis_crew.return_value = MagicMock()
        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4o-mini"
        mock_crew_system.llm = mock_llm

        # Setup mock memory system
        mock_memory = MagicMock()
        mock_memory.add_memory = MagicMock()
        mock_memory.get_memories.return_value = []
        mock_memory.search_memories.return_value = []
        mock_memory.get_memory_stats.return_value = {
            "total_memories": 0,
            "tasks_with_memory": 0,
            "agent_counts": {},
            "avg_importance": 0
        }
        mock_memory.get_context_for_task.return_value = "No previous context available."
        mock_memory.clear_memories = MagicMock()
        mock_memory.persist_path = "./data/crew_memory.json"

        # Setup module imports
        import sys
        sys.modules['app.crews'].crew_system = mock_crew_system
        sys.modules['app.memory'].crew_memory = mock_memory

        from app.main import app

        test_client = TestClient(app)

        yield test_client, mock_crew_system, mock_memory

        # Cleanup
        if 'app.main' in sys.modules:
            del sys.modules['app.main']


class TestHealthEndpoint:
    """Test suite for /health endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns correct status."""
        test_client, _, _ = client
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "ai-sdk-crewai"

    def test_health_content_type(self, client):
        """Test health endpoint returns JSON content type."""
        test_client, _, _ = client
        response = test_client.get("/health")

        assert response.headers["content-type"] == "application/json"


class TestAgentEndpoints:
    """Test suite for agent management endpoints."""

    def test_list_agents(self, client):
        """Test listing all available agents."""
        test_client, mock_crew, _ = client
        response = test_client.get("/agents")

        assert response.status_code == 200
        agents = response.json()
        assert isinstance(agents, list)
        assert len(agents) == 5
        assert "researcher" in agents

    def test_get_agent_info(self, client):
        """Test getting information about a specific agent."""
        test_client, mock_crew, _ = client
        response = test_client.get("/agents/researcher")

        assert response.status_code == 200
        data = response.json()
        assert data["role"] == "Research Specialist"
        assert data["allow_delegation"] is True

    def test_get_nonexistent_agent(self, client):
        """Test getting info for nonexistent agent."""
        test_client, mock_crew, _ = client
        mock_crew.get_agent_info.side_effect = ValueError("Agent not found")

        response = test_client.get("/agents/nonexistent")

        assert response.status_code == 404


class TestTaskExecutionEndpoints:
    """Test suite for task execution endpoints."""

    def test_execute_single_task(self, client):
        """Test executing a single task."""
        test_client, mock_crew, mock_memory = client
        response = test_client.post(
            "/tasks/execute",
            json={
                "task": "Research AI technology",
                "agent": "researcher"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["process"] == "single"
        assert data["metadata"]["agent"] == "researcher"
        assert "task_id" in data

        # Verify memory was called
        assert mock_memory.add_memory.call_count == 2  # Start and completion

    def test_execute_single_task_with_id(self, client):
        """Test executing a task with custom task ID."""
        test_client, mock_crew, mock_memory = client
        response = test_client.post(
            "/tasks/execute",
            json={
                "task": "Write content",
                "agent": "writer",
                "task_id": "custom-task-123"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "custom-task-123"

    def test_execute_single_task_with_context(self, client):
        """Test executing a task with additional context."""
        test_client, mock_crew, _ = client
        response = test_client.post(
            "/tasks/execute",
            json={
                "task": "Analyze data",
                "agent": "analyst",
                "context": "Focus on Q4 2024 trends"
            }
        )

        assert response.status_code == 200
        mock_crew.execute_sequential.assert_called_once()
        call_kwargs = mock_crew.execute_sequential.call_args.kwargs
        assert call_kwargs["context"] == "Focus on Q4 2024 trends"

    def test_execute_single_task_invalid_agent(self, client):
        """Test executing task with invalid agent."""
        test_client, mock_crew, _ = client
        mock_crew.execute_sequential.side_effect = ValueError("Agent not found")

        response = test_client.post(
            "/tasks/execute",
            json={"task": "Test task", "agent": "invalid"}
        )

        assert response.status_code == 404

    def test_execute_sequential_tasks(self, client):
        """Test executing multiple tasks sequentially."""
        test_client, mock_crew, _ = client
        response = test_client.post(
            "/tasks/sequential",
            json={
                "tasks": ["Task 1", "Task 2", "Task 3"],
                "agents": ["researcher", "writer", "reviewer"]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["process"] == "sequential"
        assert "tasks_count" in data["metadata"]

    def test_execute_sequential_without_agents(self, client):
        """Test executing sequential tasks without specifying agents."""
        test_client, mock_crew, _ = client
        response = test_client.post(
            "/tasks/sequential",
            json={"tasks": ["Task 1"]}
        )

        assert response.status_code == 200

    def test_execute_sequential_with_context(self, client):
        """Test sequential execution with context."""
        test_client, mock_crew, _ = client
        response = test_client.post(
            "/tasks/sequential",
            json={
                "tasks": ["Task 1"],
                "context": "Project context"
            }
        )

        assert response.status_code == 200
        call_kwargs = mock_crew.execute_sequential.call_args.kwargs
        assert call_kwargs["context"] == "Project context"

    def test_execute_hierarchical_tasks(self, client):
        """Test executing tasks hierarchically."""
        test_client, mock_crew, _ = client
        response = test_client.post(
            "/tasks/hierarchical",
            json={
                "manager": "planner",
                "tasks": ["Research task", "Write content"],
                "assignments": {"research": "researcher", "write": "writer"}
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["process"] == "hierarchical"
        assert data["metadata"]["manager"] == "planner"

    def test_execute_content_pipeline(self, client):
        """Test executing the content pipeline."""
        test_client, mock_crew, _ = client
        response = test_client.post(
            "/tasks/content-pipeline",
            json={"topic": "Artificial Intelligence"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["process"] == "content_pipeline"
        assert data["metadata"]["topic"] == "Artificial Intelligence"


class TestMemoryEndpoints:
    """Test suite for memory management endpoints."""

    def test_add_memory(self, client):
        """Test adding a memory entry."""
        test_client, _, mock_memory = client
        response = test_client.post(
            "/memory",
            json={
                "content": "Research findings on AI",
                "source_agent": "researcher",
                "importance": 0.9
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        mock_memory.add_memory.assert_called_once()

    def test_add_memory_with_task_id(self, client):
        """Test adding memory with task ID."""
        test_client, _, mock_memory = client
        response = test_client.post(
            "/memory",
            json={
                "content": "Task note",
                "source_agent": "writer",
                "task_id": "task-123"
            }
        )

        assert response.status_code == 200
        mock_memory.add_memory.assert_called_once()

    def test_get_memories(self, client):
        """Test retrieving memories."""
        test_client, _, mock_memory = client
        mock_memory.get_memories.return_value = [
            MagicMock(
                content="Memory 1",
                source_agent="researcher",
                task_id="task-1",
                timestamp=MagicMock(isoformat=lambda: "2024-01-01T00:00:00"),
                importance=1.0
            )
        ]

        response = test_client.get("/memory")

        assert response.status_code == 200
        memories = response.json()
        assert isinstance(memories, list)

    def test_get_memories_with_filters(self, client):
        """Test retrieving memories with filters."""
        test_client, _, mock_memory = client
        response = test_client.get(
            "/memory?agent=researcher&task_id=task-1&limit=10&min_importance=0.5"
        )

        assert response.status_code == 200
        mock_memory.get_memories.assert_called_once_with(
            agent="researcher",
            task_id="task-1",
            limit=10,
            min_importance=0.5
        )

    def test_search_memories(self, client):
        """Test searching memories."""
        test_client, _, mock_memory = client
        response = test_client.get("/memory/search?q=AI&limit=5")

        assert response.status_code == 200
        mock_memory.search_memories.assert_called_once_with("AI", 5)

    def test_get_memory_stats(self, client):
        """Test getting memory statistics."""
        test_client, _, mock_memory = client
        mock_memory.get_memory_stats.return_value = {
            "total_memories": 100,
            "tasks_with_memory": 10,
            "agent_counts": {"researcher": 50, "writer": 30},
            "avg_importance": 0.75
        }

        response = test_client.get("/memory/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_memories"] == 100
        assert data["avg_importance"] == 0.75

    def test_clear_memories(self, client):
        """Test clearing memories."""
        test_client, _, mock_memory = client
        response = test_client.delete("/memory")

        assert response.status_code == 200
        mock_memory.clear_memories.assert_called_once_with(agent=None, task_id=None)

    def test_clear_memories_with_filters(self, client):
        """Test clearing memories with filters."""
        test_client, _, mock_memory = client
        response = test_client.delete("/memory?agent=researcher")

        assert response.status_code == 200
        mock_memory.clear_memories.assert_called_once_with(agent="researcher", task_id=None)

    def test_get_task_context(self, client):
        """Test getting context for a task."""
        test_client, _, mock_memory = client
        mock_memory.get_context_for_task.return_value = "[researcher] Previous findings"

        response = test_client.get("/memory/context/task-123")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task-123"
        assert "context" in data


class TestCrewEndpoints:
    """Test suite for crew management endpoints."""

    def test_get_content_pipeline_crew(self, client):
        """Test getting content pipeline crew."""
        test_client, mock_crew, _ = client
        response = test_client.get("/crews/content-pipeline")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "workflow" in data["data"]

    def test_get_analysis_crew(self, client):
        """Test getting analysis crew."""
        test_client, mock_crew, _ = client
        response = test_client.get("/crews/analysis")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"]["type"] == "sequential"


class TestConfigEndpoint:
    """Test suite for configuration endpoint."""

    def test_get_config(self, client):
        """Test getting current configuration."""
        test_client, mock_crew, mock_memory = client
        response = test_client.get("/config")

        assert response.status_code == 200
        data = response.json()
        assert "model" in data
        assert "available_agents" in data
        assert "memory_enabled" in data
        assert data["memory_enabled"] is True
