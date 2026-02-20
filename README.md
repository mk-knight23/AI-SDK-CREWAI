# CrewAI Agent Framework SDK

A modern full-stack application with React 19 frontend and FastAPI backend.

## Tech Stack

- **Frontend**: React 19, TypeScript, Vite
- **Backend**: FastAPI, Python 3.12
- **Deployment**: Render (Docker)

## Project Structure

```
.
├── frontend/          # React 19 + TypeScript + Vite
│   ├── src/
│   │   ├── App.tsx
│   │   ├── App.css
│   │   └── main.tsx
│   ├── package.json
│   ├── tsconfig.json
│   └── Dockerfile
├── backend/           # FastAPI + Python 3.12
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── .github/
│   └── workflows/
│       └── ci.yml
└── README.md
```

## Getting Started

### Prerequisites

- Node.js 20+
- Python 3.12+
- Docker (optional)

### Development

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend will be available at http://localhost:8000

**Frontend:**
```bash
cd frontend
npm install
npm run build
```

Frontend will be available at http://localhost:3000

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/health` | GET | Health check |

### Docker

Build and run with Docker:

```bash
# Backend
docker build -t omnidesk-backend ./backend
docker run -p 8000:8000 omnidesk-backend

# Frontend
docker build -t omnidesk-frontend ./frontend
docker run -p 3000:3000 omnidesk-frontend
```

## Deployment to Render

### Web Service (Backend)

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set the root directory to `backend`
4. Use the following settings:
   - **Runtime**: Docker
   - **Port**: 8000

### Static Site (Frontend)

1. Create a new Static Site on Render
2. Connect your GitHub repository
3. Set the root directory to `frontend`
4. Use the following settings:
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `dist`

## Environment Variables

No environment variables required for basic setup.

## CI/CD

GitHub Actions workflow runs on every push to main:
- Tests backend
- Builds frontend
- Builds Docker images
