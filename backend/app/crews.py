"""CrewAI crew orchestration for multi-agent workflows."""
from crewai import Agent, Crew, Task, Process
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Optional
import os


class MultiAgentCrew:
    """Multi-agent crew orchestration with role-based delegation."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize the crew with specified LLM model.

        Args:
            model_name: The OpenAI model to use for agents
        """
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.agents = {}
        self.crews = {}
        self._init_default_agents()

    def _init_default_agents(self):
        """Initialize default set of specialized agents."""
        # Research Agent - specialized in gathering information
        self.agents["researcher"] = Agent(
            role="Research Specialist",
            goal="Conduct thorough research and gather comprehensive information on any topic",
            backstory="""You are an expert researcher with a keen eye for detail.
            You know how to find reliable sources, verify information, and
            synthesize complex topics into clear insights.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )

        # Writer Agent - specialized in content creation
        self.agents["writer"] = Agent(
            role="Content Writer",
            goal="Create engaging, well-structured content based on research findings",
            backstory="""You are a skilled writer who can transform research into
            compelling narratives. You understand structure, flow, and audience engagement.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        # Reviewer Agent - specialized in quality assurance
        self.agents["reviewer"] = Agent(
            role="Content Reviewer",
            goal="Review content for accuracy, clarity, and quality",
            backstory="""You are a meticulous editor with high standards.
            You catch errors, improve clarity, and ensure content meets quality standards.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        # Analyst Agent - specialized in data analysis
        self.agents["analyst"] = Agent(
            role="Data Analyst",
            goal="Analyze data and provide actionable insights",
            backstory="""You are a data expert who can identify patterns, trends,
            and extract meaningful insights from complex information.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )

        # Planner Agent - specialized in strategic planning
        self.agents["planner"] = Agent(
            role="Strategic Planner",
            goal="Create comprehensive plans and strategies for complex projects",
            backstory="""You are a strategic thinker who can break down complex
            problems into manageable tasks and create actionable plans.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )

    def get_agent(self, agent_name: str) -> Agent:
        """Get an agent by name.

        Args:
            agent_name: Name of the agent to retrieve

        Returns:
            The requested Agent instance

        Raises:
            ValueError: If agent doesn't exist
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found. Available agents: {list(self.agents.keys())}")
        return self.agents[agent_name]

    def list_agents(self) -> List[str]:
        """List all available agent names.

        Returns:
            List of agent names
        """
        return list(self.agents.keys())

    def add_agent(self, name: str, agent: Agent):
        """Add a custom agent to the crew.

        Args:
            name: Name identifier for the agent
            agent: Agent instance to add
        """
        self.agents[name] = agent

    def create_sequential_crew(
        self,
        tasks: List[Task],
        agents: Optional[List[str]] = None
    ) -> Crew:
        """Create a sequential crew that executes tasks one after another.

        Args:
            tasks: List of tasks to execute
            agents: Optional list of agent names (auto-assigns if None)

        Returns:
            Configured Crew instance
        """
        if agents is None:
            # Use first agent for all tasks if none specified
            agents = [list(self.agents.keys())[0]] * len(tasks)

        # Assign agents to tasks
        for task, agent_name in zip(tasks, agents):
            task.agent = self.get_agent(agent_name)

        crew = Crew(
            agents=[self.get_agent(name) for name in set(agents)],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        return crew

    def create_hierarchical_crew(
        self,
        manager_agent: str,
        tasks: List[Task],
        agent_assignments: Dict[str, str]
    ) -> Crew:
        """Create a hierarchical crew with a manager delegating to workers.

        Args:
            manager_agent: Name of the agent to act as manager
            tasks: List of tasks to execute
            agent_assignments: Dictionary mapping task descriptions to agent names

        Returns:
            Configured Crew instance
        """
        # Get manager agent
        manager = self.get_agent(manager_agent)

        # Assign agents to tasks
        for task in tasks:
            task_description = task.description
            # Find matching agent assignment
            for task_pattern, agent_name in agent_assignments.items():
                if task_pattern.lower() in task_description.lower():
                    task.agent = self.get_agent(agent_name)
                    break
            else:
                # Default to manager if no match
                task.agent = manager

        # Ensure manager can delegate
        manager.allow_delegation = True

        crew = Crew(
            agents=[manager] + [self.get_agent(name) for name in set(agent_assignments.values())],
            tasks=tasks,
            process=Process.hierarchical,
            manager_llm=self.llm,
            verbose=True
        )

        return crew

    def execute_sequential(
        self,
        task_descriptions: List[str],
        agent_names: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute tasks sequentially through agents.

        Args:
            task_descriptions: List of task descriptions
            agent_names: Optional list of agent names for each task
            context: Optional context string to provide to all tasks

        Returns:
            Dictionary with execution results
        """
        tasks = []
        for desc in task_descriptions:
            task = Task(
                description=desc,
                expected_output="A comprehensive response to the task",
                context=context
            )
            tasks.append(task)

        crew = self.create_sequential_crew(tasks, agent_names)
        result = crew.kickoff()

        return {
            "status": "completed",
            "result": str(result),
            "tasks_count": len(tasks),
            "process": "sequential"
        }

    def execute_hierarchical(
        self,
        manager_name: str,
        task_descriptions: List[str],
        agent_assignments: Dict[str, str],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute tasks with hierarchical delegation.

        Args:
            manager_name: Name of the manager agent
            task_descriptions: List of task descriptions
            agent_assignments: Mapping of task patterns to agent names
            context: Optional context for tasks

        Returns:
            Dictionary with execution results
        """
        tasks = []
        for desc in task_descriptions:
            task = Task(
                description=desc,
                expected_output="A comprehensive response to the task",
                context=context
            )
            tasks.append(task)

        crew = self.create_hierarchical_crew(manager_name, tasks, agent_assignments)
        result = crew.kickoff()

        return {
            "status": "completed",
            "result": str(result),
            "tasks_count": len(tasks),
            "process": "hierarchical",
            "manager": manager_name
        }

    def create_content_pipeline_crew(self) -> Crew:
        """Create a pre-configured crew for content creation pipeline.

        Returns:
            Crew configured for research → write → review workflow
        """
        research_task = Task(
            description="Research the given topic thoroughly and gather key information",
            expected_output="Comprehensive research notes with key findings and sources"
        )

        writing_task = Task(
            description="Write engaging content based on the research findings",
            expected_output="Well-structured, engaging content piece",
            context=research_task
        )

        review_task = Task(
            description="Review the content for accuracy, clarity, and quality",
            expected_output="Reviewed content with improvements applied",
            context=writing_task
        )

        return self.create_sequential_crew(
            [research_task, writing_task, review_task],
            ["researcher", "writer", "reviewer"]
        )

    def execute_content_pipeline(self, topic: str) -> Dict[str, Any]:
        """Execute the content creation pipeline.

        Args:
            topic: Topic to research and write about

        Returns:
            Dictionary with execution results
        """
        research_task = Task(
            description=f"Research '{topic}' thoroughly and gather key information",
            expected_output="Comprehensive research notes with key findings and sources"
        )

        writing_task = Task(
            description=f"Write engaging content about '{topic}' based on research findings",
            expected_output="Well-structured, engaging content piece",
            context=research_task
        )

        review_task = Task(
            description=f"Review the content about '{topic}' for accuracy, clarity, and quality",
            expected_output="Reviewed content with improvements applied",
            context=writing_task
        )

        crew = self.create_sequential_crew(
            [research_task, writing_task, review_task],
            ["researcher", "writer", "reviewer"]
        )

        result = crew.kickoff()

        return {
            "status": "completed",
            "result": str(result),
            "topic": topic,
            "workflow": "content_pipeline"
        }

    def create_analysis_crew(self) -> Crew:
        """Create a pre-configured crew for data analysis.

        Returns:
            Crew configured for analysis workflow
        """
        return self.create_sequential_crew(
            [],
            ["analyst"]
        )

    def get_agent_info(self, agent_name: str) -> Dict[str, str]:
        """Get information about an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary with agent information
        """
        agent = self.get_agent(agent_name)
        return {
            "role": agent.role,
            "goal": agent.goal,
            "backstory": agent.backstory,
            "allow_delegation": agent.allow_delegation,
            "verbose": agent.verbose
        }


# Global crew instance
crew_system = MultiAgentCrew()
