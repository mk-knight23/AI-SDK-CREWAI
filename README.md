# AI-SDK-CREWAI

[![AI-SDK Ecosystem](https://img.shields.io/badge/AI--SDK-ECOSYSTEM-part%20of-blue)](https://github.com/mk-knight23/AI-SDK-ECOSYSTEM)
[![CrewAI](https://img.shields.io/badge/CrewAI-0.28.0-orange)](https://github.com/joaomdmoura/crewAI)
[![React](https://img.shields.io/badge/React-19-black)](https://react.dev/)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)

> **Framework**: CrewAI (Multi-Agent Orchestration)
> **Stack**: React 19 + FastAPI + Qdrant

---

## 🎯 Project Overview

**AI-SDK-CREWAI** demonstrates production-ready multi-agent systems using CrewAI. It showcases role-based AI teams working together - Researcher, Writer, and Reviewer agents collaborating on complex tasks with RAG-powered knowledge retrieval.

### Key Features

- 👥 **Role-Based Agents** - Specialized AI agents with distinct personalities and tools
- 🔍 **RAG Integration** - Qdrant vector database for knowledge retrieval
- 📝 **Content Pipeline** - Research → Write → Review workflow
- 🔄 **Task Orchestration** - CrewAI manages agent coordination
- 📊 **Real-time Monitoring** - Watch agents collaborate in real-time

---

## 🛠 Tech Stack

### Frontend
| Technology | Version | Purpose |
|-------------|---------|---------|
| React | 19 | UI framework |
| Vite | latest | Build tool |
| MUI | v6 | Component library |
| TypeScript | 5.x | Type safety |

### Backend
| Technology | Version | Purpose |
|-------------|---------|---------|
| Python | 3.12 | Runtime |
| FastAPI | latest | API framework |
| CrewAI | 0.28+ | Agent orchestration |
| Qdrant | latest | Vector database |
| Celery | latest | Task queue |

---

## 🚀 Quick Start

```bash
# Frontend
cd frontend && npm install && npm run dev

# Backend
cd backend && pip install -r requirements.txt && python main.py
```

---

## 🔌 API Integrations

| Provider | Usage |
|----------|-------|
| OpenAI | Primary LLM |
| Anthropic | Fallback LLM |
| Groq | Speed optimization |
| HuggingFace | Embeddings |

---

## 📦 Deployment

**Render** (Backend) + **Netlify** (Frontend)

```bash
# Backend
railway up

# Frontend
netlify deploy
```

---

## 📁 Project Structure

```
AI-SDK-CREWAI/
├── frontend/         # React 19 application
├── backend/          # FastAPI + CrewAI
│   ├── crews/        # Agent crew definitions
│   └── tasks/        # Task definitions
└── README.md
```

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Part of the [AI-SDK Ecosystem](https://github.com/mk-knight23/AI-SDK-ECOSYSTEM)**
