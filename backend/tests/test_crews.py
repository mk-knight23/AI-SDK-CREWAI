"""Tests for CrewAI multi-agent crew orchestration."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.crews import MultiAgentCrew
from crewai import Agent, Task


class TestMultiAgentCrew:
    """Test suite for MultiAgentCrew class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM instance."""
        mock = MagicMock()
        mock.model_name = "gpt-4o-mini"
        mock.temperature = 0.7
        return mock

    @pytest.fixture
    def crew(self, mock_llm):
        """Create a MultiAgentCrew instance with mocked dependencies."""
        with patch('app.crews.ChatOpenAI', return_value=mock_llm):
            with patch('app.crews.Agent') as mock_agent_class:
                # Setup mock agent instances
                mock_agent = MagicMock()
                mock_agent.role = "Test Role"
                mock_agent.goal = "Test Goal"
                mock_agent.backstory = "Test Backstory"
                mock_agent.allow_delegation = True
                mock_agent.verbose = True
                mock_agent_class.return_value = mock_agent

                test_crew = MultiAgentCrew()

                # Store the mock agent for assertions
                test_crew._mock_agent = mock_agent

                yield test_crew, mock_agent, mock_llm

    def test_init_creates_default_agents(self, crew):
        """Test that initialization creates all default agents."""
        test_crew, _, _ = crew

        expected_agents = ["researcher", "writer", "reviewer", "analyst", "planner"]
        for agent_name in expected_agents:
            assert agent_name in test_crew.agents

    def test_init_with_custom_model(self, mock_llm):
        """Test initialization with custom model name."""
        with patch('app.crews.ChatOpenAI', return_value=mock_llm) as mock_llm_class:
            with patch('app.crews.Agent', return_value=MagicMock()):
                MultiAgentCrew(model_name="gpt-4")

                mock_llm_class.assert_called_once_with(model="gpt-4", temperature=0.7)

    def test_get_existing_agent(self, crew):
        """Test getting an existing agent by name."""
        test_crew, mock_agent, _ = crew

        result = test_crew.get_agent("researcher")

        assert result == mock_agent

    def test_get_nonexistent_agent_raises_error(self, crew):
        """Test that getting a nonexistent agent raises ValueError."""
        test_crew, _, _ = crew

        with pytest.raises(ValueError) as exc_info:
            test_crew.get_agent("nonexistent")

        assert "nonexistent" in str(exc_info.value)
        assert "researcher" in str(exc_info.value)

    def test_list_agents(self, crew):
        """Test listing all available agents."""
        test_crew, _, _ = crew

        agents = test_crew.list_agents()

        expected = ["researcher", "writer", "reviewer", "analyst", "planner"]
        assert sorted(agents) == sorted(expected)

    def test_add_custom_agent(self, crew):
        """Test adding a custom agent."""
        test_crew, _, _ = crew

        custom_agent = MagicMock()
        custom_agent.role = "Custom Agent"

        test_crew.add_agent("custom", custom_agent)

        assert "custom" in test_crew.agents
        assert test_crew.agents["custom"] == custom_agent

    def test_create_sequential_crew_default_agent(self, crew):
        """Test creating sequential crew with default agent assignment."""
        test_crew, mock_agent, _ = crew

        task1 = MagicMock()
        task1.description = "Task 1"
        task2 = MagicMock()
        task2.description = "Task 2"

        with patch('app.crews.Crew') as mock_crew_class:
            mock_crew = MagicMock()
            mock_crew_class.return_value = mock_crew

            result = test_crew.create_sequential_crew([task1, task2])

            # Verify crew was created
            mock_crew_class.assert_called_once()
            call_kwargs = mock_crew_class.call_args[1]

            assert call_kwargs["process"] == "sequential"
            assert call_kwargs["verbose"] is True

    def test_create_sequential_crew_with_agents(self, crew):
        """Test creating sequential crew with specified agents."""
        test_crew, mock_agent, _ = crew

        task1 = MagicMock()
        task1.description = "Task 1"

        with patch('app.crews.Crew', return_value=MagicMock()):
            result = test_crew.create_sequential_crew(
                [task1],
                agents=["researcher"]
            )

            # Verify agent was assigned to task
            assert task1.agent == mock_agent

    def test_create_hierarchical_crew(self, crew):
        """Test creating hierarchical crew with manager."""
        test_crew, mock_agent, _ = crew

        task1 = MagicMock()
        task1.description = "Analyze the data"

        with patch('app.crews.Crew', return_value=MagicMock()):
            result = test_crew.create_hierarchical_crew(
                manager_agent="planner",
                tasks=[task1],
                agent_assignments={"analyze": "analyst"}
            )

            # Verify manager delegation is enabled
            assert test_crew.agents["planner"].allow_delegation is True

    def test_execute_sequential(self, crew):
        """Test executing tasks sequentially."""
        test_crew, _, _ = crew

        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Execution result"

        with patch('app.crews.Crew', return_value=mock_crew):
            result = test_crew.execute_sequential(
                task_descriptions=["Task 1", "Task 2"],
                agent_names=["researcher", "writer"]
            )

            assert result["status"] == "completed"
            assert result["result"] == "Execution result"
            assert result["tasks_count"] == 2
            assert result["process"] == "sequential"

    def test_execute_hierarchical(self, crew):
        """Test executing tasks with hierarchical delegation."""
        test_crew, _, _ = crew

        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Hierarchical result"

        with patch('app.crews.Crew', return_value=mock_crew):
            result = test_crew.execute_hierarchical(
                manager_name="planner",
                task_descriptions=["Research task"],
                agent_assignments={"research": "researcher"}
            )

            assert result["status"] == "completed"
            assert result["result"] == "Hierarchical result"
            assert result["process"] == "hierarchical"
            assert result["manager"] == "planner"

    def test_create_content_pipeline_crew(self, crew):
        """Test creating pre-configured content pipeline crew."""
        test_crew, _, _ = crew

        with patch('app.crews.Crew', return_value=MagicMock()):
            result = test_crew.create_content_pipeline_crew()

            # Verify Crew was called
            assert result is not None

    def test_execute_content_pipeline(self, crew):
        """Test executing the content pipeline workflow."""
        test_crew, _, _ = crew

        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Content created successfully"

        with patch('app.crews.Crew', return_value=mock_crew):
            result = test_crew.execute_content_pipeline("AI Technology")

            assert result["status"] == "completed"
            assert result["topic"] == "AI Technology"
            assert result["workflow"] == "content_pipeline"
            assert result["result"] == "Content created successfully"

    def test_create_analysis_crew(self, crew):
        """Test creating analysis crew."""
        test_crew, _, _ = crew

        with patch('app.crews.Crew', return_value=MagicMock()):
            result = test_crew.create_analysis_crew()

            assert result is not None

    def test_get_agent_info(self, crew):
        """Test getting agent information."""
        test_crew, mock_agent, _ = crew

        info = test_crew.get_agent_info("researcher")

        assert info["role"] == "Test Role"
        assert info["goal"] == "Test Goal"
        assert info["backstory"] == "Test Backstory"
        assert info["allow_delegation"] is True
        assert info["verbose"] is True

    def test_execute_with_context(self, crew):
        """Test executing tasks with additional context."""
        test_crew, _, _ = crew

        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Result with context"

        with patch('app.crews.Crew', return_value=mock_crew):
            result = test_crew.execute_sequential(
                task_descriptions=["Task 1"],
                context="Additional context information"
            )

            assert result["status"] == "completed"
            assert result["result"] == "Result with context"


class TestAgentRoles:
    """Test suite for agent role definitions."""

    @pytest.fixture
    def crew(self):
        """Create crew instance."""
        with patch('app.crews.ChatOpenAI', return_value=MagicMock()):
            with patch('app.crews.Agent', return_value=MagicMock()):
                return MultiAgentCrew()

    def test_researcher_agent_configured(self, crew):
        """Test researcher agent has correct configuration."""
        with patch('app.crews.Agent') as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            crew._init_default_agents()

            # Check Agent was called for researcher
            calls = mock_agent_class.call_args_list
            researcher_call = [c for c in calls if "research" in str(c).lower()]

            assert len(researcher_call) > 0

    def test_writer_agent_configured(self, crew):
        """Test writer agent has correct configuration."""
        with patch('app.crews.Agent') as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            crew._init_default_agents()

            calls = mock_agent_class.call_args_list
            writer_call = [c for c in calls if "writer" in str(c).lower() or "content" in str(c).lower()]

            assert len(writer_call) > 0

    def test_reviewer_agent_configured(self, crew):
        """Test reviewer agent has correct configuration."""
        with patch('app.crews.Agent') as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent

            crew._init_default_agents()

            calls = mock_agent_class.call_args_list
            reviewer_call = [c for c in calls if "reviewer" in str(c).lower() or "review" in str(c).lower()]

            assert len(reviewer_call) > 0


class TestTaskExecution:
    """Test suite for task execution patterns."""

    @pytest.fixture
    def crew(self):
        """Create crew instance."""
        with patch('app.crews.ChatOpenAI', return_value=MagicMock()):
            with patch('app.crews.Agent', return_value=MagicMock()):
                test_crew = MultiAgentCrew()
                return test_crew

    def test_empty_task_list(self, crew):
        """Test executing empty task list."""
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "No tasks executed"

        with patch('app.crews.Crew', return_value=mock_crew):
            result = crew.execute_sequential([])

            assert result["tasks_count"] == 0

    def test_single_task_execution(self, crew):
        """Test executing a single task."""
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Single task done"

        with patch('app.crews.Crew', return_value=mock_crew):
            result = crew.execute_sequential(
                task_descriptions=["Single task"]
            )

            assert result["tasks_count"] == 1
            assert result["status"] == "completed"

    def test_multiple_task_execution(self, crew):
        """Test executing multiple tasks."""
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Multiple tasks done"

        with patch('app.crews.Crew', return_value=mock_crew):
            result = crew.execute_sequential(
                task_descriptions=["Task 1", "Task 2", "Task 3", "Task 4", "Task 5"]
            )

            assert result["tasks_count"] == 5
