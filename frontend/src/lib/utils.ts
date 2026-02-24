import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000"

export async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`

  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }))
    throw new Error(error.detail || "API request failed")
  }

  return response.json()
}

export const api = {
  // Health check
  health: () => apiRequest<{ status: string; service: string }>("/health"),

  // Agents
  listAgents: () => apiRequest<string[]>("/agents"),
  getAgent: (name: string) =>
    apiRequest<{
      role: string
      goal: string
      backstory: string
      allow_delegation: boolean
    }>(`/agents/${name}`),

  // Tasks
  executeTask: (data: { task: string; agent: string; task_id?: string; context?: string }) =>
    apiRequest<{
      status: string
      result: string
      task_id: string
      process: string
      metadata: Record<string, unknown>
    }>("/tasks/execute", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  executeSequential: (data: {
    tasks: string[]
    agents?: string[]
    context?: string
  }) =>
    apiRequest<{
      status: string
      result: string
      process: string
      metadata: { tasks_count: number }
    }>("/tasks/sequential", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  executeHierarchical: (data: {
    manager: string
    tasks: string[]
    assignments: Record<string, string>
    context?: string
  }) =>
    apiRequest<{
      status: string
      result: string
      process: string
      metadata: { manager: string; tasks_count: number }
    }>("/tasks/hierarchical", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  executeContentPipeline: (data: { topic: string }) =>
    apiRequest<{
      status: string
      result: string
      process: string
      metadata: { topic: string }
    }>("/tasks/content-pipeline", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  // Memory
  addMemory: (data: {
    content: string
    source_agent: string
    task_id?: string
    importance?: number
  }) =>
    apiRequest<{ status: string; message: string }>("/memory", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  getMemories: (params?: {
    agent?: string
    task_id?: string
    limit?: number
    min_importance?: number
  }) => {
    const searchParams = new URLSearchParams()
    if (params?.agent) searchParams.set("agent", params.agent)
    if (params?.task_id) searchParams.set("task_id", params.task_id)
    if (params?.limit) searchParams.set("limit", params.limit.toString())
    if (params?.min_importance)
      searchParams.set("min_importance", params.min_importance.toString())

    return apiRequest<
      Array<{
        content: string
        source_agent: string
        task_id: string | null
        timestamp: string
        importance: number
      }>
    >(`/memory?${searchParams.toString()}`)
  },

  searchMemories: (q: string, limit = 5) =>
    apiRequest<
      Array<{
        content: string
        source_agent: string
        task_id: string | null
        timestamp: string
        importance: number
      }>
    >(`/memory/search?q=${encodeURIComponent(q)}&limit=${limit}`),

  getMemoryStats: () =>
    apiRequest<{
      total_memories: number
      tasks_with_memory: number
      agent_counts: Record<string, number>
      avg_importance: number
    }>("/memory/stats"),

  clearMemories: (params?: { agent?: string; task_id?: string }) => {
    const searchParams = new URLSearchParams()
    if (params?.agent) searchParams.set("agent", params.agent)
    if (params?.task_id) searchParams.set("task_id", params.task_id)

    return apiRequest<{ status: string; message: string }>(
      `/memory?${searchParams.toString()}`,
      { method: "DELETE" }
    )
  },

  getTaskContext: (taskId: string) =>
    apiRequest<{ task_id: string; context: string }>(
      `/memory/context/${taskId}`
    ),

  // Config
  getConfig: () =>
    apiRequest<{
      model: string
      available_agents: string[]
      memory_enabled: boolean
      persistence_path: string
    }>("/config"),
}
