# Deployment Guide

**AI-SDK-CREWAI** - Production Deployment Instructions

---

## Environment Setup

### Required Environment Variables

```bash
# API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-openai-...

# Database
DATABASE_URL=postgresql://user:pass@host:5432/dbname
REDIS_URL=redis://host:6379

# Application
NODE_ENV=production
API_BASE_URL=https://api.crewai.com
```

---

## Platform Deployments

### Vercel (Frontend)

```bash
cd frontend
vercel --prod
```

### Railway (Backend)

```bash
cd backend
railway up
```

### Fly.io (Full Stack)

```bash
fly deploy
```

---

## Docker Deployment

```bash
docker-compose up -d
```

---

## Health Check

```bash
curl https://api.crewai.com/health
```

---

For more details, see the project README.
