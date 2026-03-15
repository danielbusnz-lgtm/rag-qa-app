from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from .models import IngestURLRequest, IngestTextRequest, QueryRequest, IngestResponse
from .ingestion import DocumentIngester
from .retrieval import VectorStore
from .chain import RAGChain
import os

load_dotenv()

vector_store = VectorStore()
ingester = DocumentIngester()
rag_chain = RAGChain(vector_store)

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("./chroma_db", exist_ok=True)
    yield

app = FastAPI(title="RAG Q&A API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest/pdf", response_model = IngestResponse)
async def ingest_pdf(
        file: UploadFile = File(...),
        collection_name: str = "default"
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail = "only pdf files supported")
    content = await file.read()
    chunks = ingester.ingest_pdf(content, file.filename)
    added = vector_store.add_documents(chunks, collection_name)
    return IngestResponse(
        message f"Successfully ingested{file.filename}",
        chunks_added =added,
        collection_name = collection_name
    )
