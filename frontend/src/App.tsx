import { useState, useEffect } from "react"
import { Button } from "./components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card"
import { Input } from "./components/ui/input"
import { Textarea } from "./components/ui/textarea"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs"
import { Badge } from "./components/ui/badge"
import { api } from "./lib/utils"
import {
  Users,
  Zap,
  Brain,
  FileText,
  Search,
  Database,
  Loader2,
  CheckCircle2,
  AlertCircle,
} from "lucide-react"

interface Agent {
  name: string
  role: string
  goal: string
  backstory: string
  allow_delegation: boolean
}

interface Memory {
  content: string
  source_agent: string
  task_id: string | null
  timestamp: string
  importance: number
}

interface TaskResult {
  status: string
  result: string
  task_id?: string
  process: string
  metadata: Record<string, unknown>
}

interface MemoryStats {
  total_memories: number
  tasks_with_memory: number
  agent_counts: Record<string, number>
  avg_importance: number
}

const agentIcons: Record<string, React.ReactNode> = {
  researcher: <Search className="h-5 w-5" />,
  writer: <FileText className="h-5 w-5" />,
  reviewer: <CheckCircle2 className="h-5 w-5" />,
  analyst: <Brain className="h-5 w-5" />,
  planner: <Zap className="h-5 w-5" />,
}

function App() {
  const [agents, setAgents] = useState<string[]>([])
  const [agentDetails, setAgentDetails] = useState<Record<string, Agent>>({})
  const [memories, setMemories] = useState<Memory[]>([])
  const [memoryStats, setMemoryStats] = useState<MemoryStats | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [taskResult, setTaskResult] = useState<TaskResult | null>(null)

  // Single task form
  const [singleTask, setSingleTask] = useState("")
  const [selectedAgent, setSelectedAgent] = useState("researcher")

  // Sequential tasks form
  const [sequentialTasks, setSequentialTasks] = useState("")
  const [sequentialAgents, setSequentialAgents] = useState("")

  // Content pipeline form
  const [topic, setTopic] = useState("")

  // Hierarchical form
  const [manager, setManager] = useState("planner")
  const [hierarchicalTasks, setHierarchicalTasks] = useState("")
  const [hierarchicalAssignments, setHierarchicalAssignments] = useState("")

  // Memory search
  const [searchQuery, setSearchQuery] = useState("")

  useEffect(() => {
    loadAgents()
    loadMemories()
    loadMemoryStats()
  }, [])

  const loadAgents = async () => {
    try {
      const agentList = await api.listAgents()
      setAgents(agentList)

      // Load details for each agent
      const details: Record<string, Agent> = {}
      for (const agent of agentList) {
        const info = await api.getAgent(agent)
        details[agent] = { name: agent, ...info }
      }
      setAgentDetails(details)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load agents")
    }
  }

  const loadMemories = async () => {
    try {
      const data = await api.getMemories({ limit: 20 })
      setMemories(data)
    } catch (err) {
      console.error("Failed to load memories:", err)
    }
  }

  const loadMemoryStats = async () => {
    try {
      const stats = await api.getMemoryStats()
      setMemoryStats(stats)
    } catch (err) {
      console.error("Failed to load memory stats:", err)
    }
  }

  const handleSingleTask = async () => {
    if (!singleTask.trim()) return

    setLoading(true)
    setError(null)
    setTaskResult(null)

    try {
      const result = await api.executeTask({
        task: singleTask,
        agent: selectedAgent,
      })
      setTaskResult(result)
      loadMemories()
      loadMemoryStats()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Task execution failed")
    } finally {
      setLoading(false)
    }
  }

  const handleSequentialTasks = async () => {
    const tasks = sequentialTasks
      .split("\n")
      .map((t) => t.trim())
      .filter(Boolean)

    if (tasks.length === 0) return

    setLoading(true)
    setError(null)
    setTaskResult(null)

    try {
      const agents = sequentialAgents
        .split("\n")
        .map((a) => a.trim())
        .filter(Boolean)

      const result = await api.executeSequential({
        tasks,
        agents: agents.length > 0 ? agents : undefined,
      })
      setTaskResult(result)
      loadMemories()
      loadMemoryStats()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sequential execution failed")
    } finally {
      setLoading(false)
    }
  }

  const handleContentPipeline = async () => {
    if (!topic.trim()) return

    setLoading(true)
    setError(null)
    setTaskResult(null)

    try {
      const result = await api.executeContentPipeline({ topic })
      setTaskResult(result)
      loadMemories()
      loadMemoryStats()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Pipeline execution failed")
    } finally {
      setLoading(false)
    }
  }

  const handleHierarchicalTasks = async () => {
    const tasks = hierarchicalTasks
      .split("\n")
      .map((t) => t.trim())
      .filter(Boolean)

    if (tasks.length === 0) return

    setLoading(true)
    setError(null)
    setTaskResult(null)

    try {
      const assignments: Record<string, string> = {}
      hierarchicalAssignments.split("\n").forEach((line) => {
        const [pattern, agent] = line.split(":").map((s) => s.trim())
        if (pattern && agent) {
          assignments[pattern] = agent
        }
      })

      const result = await api.executeHierarchical({
        manager,
        tasks,
        assignments,
      })
      setTaskResult(result)
      loadMemories()
      loadMemoryStats()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Hierarchical execution failed")
    } finally {
      setLoading(false)
    }
  }

  const handleSearchMemories = async () => {
    if (!searchQuery.trim()) {
      loadMemories()
      return
    }

    try {
      const results = await api.searchMemories(searchQuery, 20)
      setMemories(results)
    } catch (err) {
      console.error("Search failed:", err)
    }
  }

  const handleClearMemories = async () => {
    try {
      await api.clearMemories()
      loadMemories()
      loadMemoryStats()
    } catch (err) {
      console.error("Failed to clear memories:", err)
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Users className="h-8 w-8 text-primary" />
              <div>
                <h1 className="text-2xl font-bold">AI-SDK-CREWAI</h1>
                <p className="text-sm text-muted-foreground">
                  Multi-Agent Crew Orchestration Platform
                </p>
              </div>
            </div>
            <Badge variant="secondary" className="gap-1">
              <CheckCircle2 className="h-3 w-3" />
              Connected
            </Badge>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        {error && (
          <Card className="mb-6 border-destructive">
            <CardContent className="pt-6">
              <div className="flex items-center gap-2 text-destructive">
                <AlertCircle className="h-5 w-5" />
                <p>{error}</p>
              </div>
            </CardContent>
          </Card>
        )}

        <Tabs defaultValue="agents" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="agents">Agents</TabsTrigger>
            <TabsTrigger value="single">Single Task</TabsTrigger>
            <TabsTrigger value="workflows">Workflows</TabsTrigger>
            <TabsTrigger value="memory">Memory</TabsTrigger>
            <TabsTrigger value="results">Results</TabsTrigger>
          </TabsList>

          {/* Agents Tab */}
          <TabsContent value="agents" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {Object.values(agentDetails).map((agent) => (
                <Card key={agent.name}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="capitalize flex items-center gap-2">
                        {agentIcons[agent.name] || <Users className="h-5 w-5" />}
                        {agent.name}
                      </CardTitle>
                      <Badge
                        variant={agent.allow_delegation ? "default" : "secondary"}
                      >
                        {agent.allow_delegation ? "Can Delegate" : "No Delegation"}
                      </Badge>
                    </div>
                    <CardDescription>{agent.role}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <div>
                      <p className="text-sm font-medium">Goal</p>
                      <p className="text-sm text-muted-foreground">{agent.goal}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Backstory</p>
                      <p className="text-sm text-muted-foreground line-clamp-2">
                        {agent.backstory}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Single Task Tab */}
          <TabsContent value="single">
            <Card>
              <CardHeader>
                <CardTitle>Execute Single Task</CardTitle>
                <CardDescription>
                  Assign a task to a specific agent
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium">Select Agent</label>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {agents.map((agent) => (
                      <Button
                        key={agent}
                        variant={selectedAgent === agent ? "default" : "outline"}
                        size="sm"
                        onClick={() => setSelectedAgent(agent)}
                      >
                        {agentIcons[agent]}
                        {agent}
                      </Button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="text-sm font-medium">Task Description</label>
                  <Textarea
                    placeholder="Describe the task you want the agent to perform..."
                    value={singleTask}
                    onChange={(e) => setSingleTask(e.target.value)}
                    rows={4}
                  />
                </div>
                <Button onClick={handleSingleTask} disabled={loading}>
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Executing...
                    </>
                  ) : (
                    "Execute Task"
                  )}
                </Button>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Workflows Tab */}
          <TabsContent value="workflows" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Sequential Tasks</CardTitle>
                <CardDescription>
                  Execute multiple tasks in sequence (one per line)
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium">Tasks</label>
                  <Textarea
                    placeholder="Task 1&#10;Task 2&#10;Task 3"
                    value={sequentialTasks}
                    onChange={(e) => setSequentialTasks(e.target.value)}
                    rows={4}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">
                    Agents (optional, one per line)
                  </label>
                  <Textarea
                    placeholder="researcher&#10;writer&#10;reviewer"
                    value={sequentialAgents}
                    onChange={(e) => setSequentialAgents(e.target.value)}
                    rows={3}
                  />
                </div>
                <Button
                  onClick={handleSequentialTasks}
                  disabled={loading}
                  variant="secondary"
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Executing...
                    </>
                  ) : (
                    "Execute Sequentially"
                  )}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Content Pipeline</CardTitle>
                <CardDescription>
                  Research → Write → Review workflow
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium">Topic</label>
                  <Input
                    placeholder="e.g., The Future of AI"
                    value={topic}
                    onChange={(e) => setTopic(e.target.value)}
                  />
                </div>
                <Button
                  onClick={handleContentPipeline}
                  disabled={loading}
                  variant="secondary"
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Running Pipeline...
                    </>
                  ) : (
                    "Execute Content Pipeline"
                  )}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Hierarchical Delegation</CardTitle>
                <CardDescription>
                  Manager agent delegates tasks to specialists
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium">Manager Agent</label>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {agents.map((agent) => (
                      <Button
                        key={agent}
                        variant={manager === agent ? "default" : "outline"}
                        size="sm"
                        onClick={() => setManager(agent)}
                      >
                        {agentIcons[agent]}
                        {agent}
                      </Button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="text-sm font-medium">Tasks</label>
                  <Textarea
                    placeholder="Research AI trends&#10;Write summary&#10;Analyze data"
                    value={hierarchicalTasks}
                    onChange={(e) => setHierarchicalTasks(e.target.value)}
                    rows={4}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">
                    Assignments (pattern: agent)
                  </label>
                  <Textarea
                    placeholder="research: researcher&#10;write: writer&#10;analyze: analyst"
                    value={hierarchicalAssignments}
                    onChange={(e) => setHierarchicalAssignments(e.target.value)}
                    rows={3}
                  />
                </div>
                <Button
                  onClick={handleHierarchicalTasks}
                  disabled={loading}
                  variant="secondary"
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Executing...
                    </>
                  ) : (
                    "Execute Hierarchically"
                  )}
                </Button>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Memory Tab */}
          <TabsContent value="memory" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Memory Statistics</CardTitle>
              </CardHeader>
              <CardContent>
                {memoryStats && (
                  <div className="grid gap-4 md:grid-cols-4">
                    <div>
                      <p className="text-2xl font-bold">{memoryStats.total_memories}</p>
                      <p className="text-sm text-muted-foreground">Total Memories</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold">{memoryStats.tasks_with_memory}</p>
                      <p className="text-sm text-muted-foreground">Tasks with Memory</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold">
                        {memoryStats.avg_importance.toFixed(2)}
                      </p>
                      <p className="text-sm text-muted-foreground">Avg Importance</p>
                    </div>
                    <Button variant="destructive" size="sm" onClick={handleClearMemories}>
                      Clear All
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Memory Search</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-2">
                  <Input
                    placeholder="Search memories..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSearchMemories()}
                  />
                  <Button onClick={handleSearchMemories}>
                    <Search className="h-4 w-4" />
                  </Button>
                </div>
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {memories.map((memory, index) => (
                    <Card key={index}>
                      <CardContent className="pt-4">
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <Badge variant="outline">{memory.source_agent}</Badge>
                              <span className="text-xs text-muted-foreground">
                                {new Date(memory.timestamp).toLocaleString()}
                              </span>
                            </div>
                            <p className="text-sm">{memory.content}</p>
                          </div>
                          <Badge
                            variant={
                              memory.importance > 0.7
                                ? "default"
                                : memory.importance > 0.4
                                ? "secondary"
                                : "outline"
                            }
                          >
                            {memory.importance.toFixed(1)}
                          </Badge>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                  {memories.length === 0 && (
                    <p className="text-center text-muted-foreground py-8">
                      No memories found
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Results Tab */}
          <TabsContent value="results">
            <Card>
              <CardHeader>
                <CardTitle>Task Execution Results</CardTitle>
              </CardHeader>
              <CardContent>
                {taskResult ? (
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="h-5 w-5 text-green-500" />
                      <Badge variant="default">{taskResult.status}</Badge>
                      <Badge variant="outline">{taskResult.process}</Badge>
                    </div>
                    <div className="rounded-lg bg-muted p-4">
                      <pre className="whitespace-pre-wrap text-sm">
                        {taskResult.result}
                      </pre>
                    </div>
                    {Object.keys(taskResult.metadata).length > 0 && (
                      <div>
                        <p className="text-sm font-medium mb-2">Metadata</p>
                        <div className="rounded-lg bg-muted p-4">
                          <pre className="text-sm">
                            {JSON.stringify(taskResult.metadata, null, 2)}
                          </pre>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-center text-muted-foreground py-8">
                    No results yet. Execute a task to see results here.
                  </p>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}

export default App
