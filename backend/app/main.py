"""FastAPI application for CrewAI multi-agent orchestration."""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

from app.crews import crew_system
from app.memory import crew_memory, TaskExecutionContext

load_dotenv()

app = FastAPI(
    title="AI-SDK-CREWAI",
    description="Multi-agent crew orchestration with CrewAI",
    version="1.0.0"
)


# Request/Response Models
class AgentInfo(BaseModel):
    """Information about an agent."""
    role: str
    goal: str
    backstory: str
    allow_delegation: bool
    verbose: bool


class TaskRequest(BaseModel):
    """Request to execute a task."""
    task: str = Field(description="Task description")
    agent: str = Field(description="Agent to execute the task")
    context: Optional[str] = Field(None, description="Additional context")
    task_id: Optional[str] = Field(None, description="Task ID for tracking")


class SequentialTasksRequest(BaseModel):
    """Request to execute tasks sequentially."""
    tasks: List[str] = Field(description="List of task descriptions")
    agents: Optional[List[str]] = Field(None, description="Agent for each task")
    context: Optional[str] = Field(None, description="Additional context")


class HierarchicalTasksRequest(BaseModel):
    """Request to execute tasks hierarchically."""
    manager: str = Field(description="Manager agent name")
    tasks: List[str] = Field(description="List of task descriptions")
    assignments: Dict[str, str] = Field(description="Task pattern to agent mapping")
    context: Optional[str] = Field(None, description="Additional context")


class ContentPipelineRequest(BaseModel):
    """Request to execute content pipeline."""
    topic: str = Field(description="Topic to research and write about")


class MemoryEntry(BaseModel):
    """A memory entry."""
    content: str
    source_agent: str
    task_id: Optional[str] = None
    importance: float = 1.0


class TaskExecutionResponse(BaseModel):
    """Response from task execution."""
    status: str
    result: str
    task_id: Optional[str] = None
    process: str
    metadata: Dict[str, Any] = {}


class CrewResponse(BaseModel):
    """Response from crew operations."""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


# Health Check
@app.get("/health", response_model=Dict[str, str])
def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ai-sdk-crewai"}


# Agent Management
@app.get("/agents", response_model=List[str])
def list_agents():
    """List all available agents."""
    return crew_system.list_agents()


@app.get("/agents/{agent_name}", response_model=AgentInfo)
def get_agent_info(agent_name: str):
    """Get information about a specific agent."""
    try:
        return crew_system.get_agent_info(agent_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Task Execution
@app.post("/tasks/execute", response_model=TaskExecutionResponse)
def execute_single_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Execute a single task with an agent."""
    try:
        task_id = request.task_id or f"task-{hash(request.task) % 10000}"

        # Record task start in memory
        crew_memory.add_memory(
            content=f"Task: {request.task}",
            source_agent=request.agent,
            task_id=task_id,
            importance=1.0
        )

        # Execute the task
        result = crew_system.execute_sequential(
            task_descriptions=[request.task],
            agent_names=[request.agent],
            context=request.context
        )

        # Record completion
        crew_memory.add_memory(
            content=f"Task completed: {request.task}",
            source_agent=request.agent,
            task_id=task_id,
            importance=0.8
        )

        return TaskExecutionResponse(
            status=result["status"],
            result=result["result"],
            task_id=task_id,
            process="single",
            metadata={"agent": request.agent}
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


@app.post("/tasks/sequential", response_model=TaskExecutionResponse)
def execute_sequential_tasks(request: SequentialTasksRequest):
    """Execute multiple tasks sequentially."""
    try:
        result = crew_system.execute_sequential(
            task_descriptions=request.tasks,
            agent_names=request.agents,
            context=request.context
        )

        return TaskExecutionResponse(
            status=result["status"],
            result=result["result"],
            process="sequential",
            metadata={"tasks_count": result["tasks_count"]}
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


@app.post("/tasks/hierarchical", response_model=TaskExecutionResponse)
def execute_hierarchical_tasks(request: HierarchicalTasksRequest):
    """Execute tasks with hierarchical delegation."""
    try:
        result = crew_system.execute_hierarchical(
            manager_name=request.manager,
            task_descriptions=request.tasks,
            agent_assignments=request.assignments,
            context=request.context
        )

        return TaskExecutionResponse(
            status=result["status"],
            result=result["result"],
            process="hierarchical",
            metadata={"manager": request.manager, "tasks_count": result["tasks_count"]}
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


@app.post("/tasks/content-pipeline", response_model=TaskExecutionResponse)
def execute_content_pipeline(request: ContentPipelineRequest):
    """Execute the research → write → review content pipeline."""
    try:
        result = crew_system.execute_content_pipeline(request.topic)

        return TaskExecutionResponse(
            status=result["status"],
            result=result["result"],
            process="content_pipeline",
            metadata={"topic": request.topic}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


# Memory Management
@app.post("/memory", response_model=CrewResponse)
def add_memory(entry: MemoryEntry):
    """Add a memory entry."""
    crew_memory.add_memory(
        content=entry.content,
        source_agent=entry.source_agent,
        task_id=entry.task_id,
        importance=entry.importance
    )

    return CrewResponse(
        status="success",
        message="Memory added successfully"
    )


@app.get("/memory", response_model=List[Dict[str, Any]])
def get_memories(
    agent: Optional[str] = None,
    task_id: Optional[str] = None,
    limit: Optional[int] = None,
    min_importance: float = 0.0
):
    """Retrieve memories with optional filters."""
    memories = crew_memory.get_memories(
        agent=agent,
        task_id=task_id,
        limit=limit,
        min_importance=min_importance
    )

    return [
        {
            "content": m.content,
            "source_agent": m.source_agent,
            "task_id": m.task_id,
            "timestamp": m.timestamp.isoformat(),
            "importance": m.importance
        }
        for m in memories
    ]


@app.get("/memory/search", response_model=List[Dict[str, Any]])
def search_memories(q: str, limit: int = 5):
    """Search memories by content."""
    memories = crew_memory.search_memories(q, limit)

    return [
        {
            "content": m.content,
            "source_agent": m.source_agent,
            "task_id": m.task_id,
            "timestamp": m.timestamp.isoformat(),
            "importance": m.importance
        }
        for m in memories
    ]


@app.get("/memory/stats", response_model=Dict[str, Any])
def get_memory_stats():
    """Get memory statistics."""
    return crew_memory.get_memory_stats()


@app.delete("/memory", response_model=CrewResponse)
def clear_memories(
    agent: Optional[str] = None,
    task_id: Optional[str] = None
):
    """Clear memories with optional filters."""
    crew_memory.clear_memories(agent=agent, task_id=task_id)

    return CrewResponse(
        status="success",
        message="Memories cleared successfully"
    )


@app.get("/memory/context/{task_id}", response_model=Dict[str, str])
def get_task_context(task_id: str):
    """Get formatted context for a task."""
    context = crew_memory.get_context_for_task(task_id)

    return {"task_id": task_id, "context": context}


# Crew Management
@app.get("/crews/content-pipeline", response_model=CrewResponse)
def get_content_pipeline_crew():
    """Get the content pipeline crew configuration."""
    crew = crew_system.create_content_pipeline_crew()

    return CrewResponse(
        status="success",
        message="Content pipeline crew created",
        data={"type": "sequential", "workflow": "research → write → review"}
    )


@app.get("/crews/analysis", response_model=CrewResponse)
def get_analysis_crew():
    """Get the analysis crew configuration."""
    crew = crew_system.create_analysis_crew()

    return CrewResponse(
        status="success",
        message="Analysis crew created",
        data={"type": "sequential", "workflow": "data analysis"}
    )


# Configuration
@app.get("/config", response_model=Dict[str, Any])
def get_config():
    """Get current configuration."""
    return {
        "model": crew_system.llm.model_name if hasattr(crew_system, 'llm') else "gpt-4o-mini",
        "available_agents": crew_system.list_agents(),
        "memory_enabled": True,
        "persistence_path": crew_memory.persist_path
    }
