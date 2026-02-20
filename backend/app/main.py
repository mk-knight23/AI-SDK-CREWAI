from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import rag_system
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="OmniDesk - Enterprise RAG")


class DocumentInput(BaseModel):
    documents: list[str]


class QueryInput(BaseModel):
    question: str


class QueryOutput(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "healthy", "service": "omni-desk"}


@app.post("/documents")
def add_documents(input: DocumentInput):
    rag_system.add_documents(input.documents)
    return {"status": "success", "document_count": len(input.documents)}


@app.post("/query", response_model=QueryOutput)
def query(input: QueryInput):
    result = rag_system.query(input.question)
    return QueryOutput(**result)
