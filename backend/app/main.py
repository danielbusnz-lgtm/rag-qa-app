"""FastAPI backend for a RAG Q&A application.

Exposes endpoints for ingesting documents (PDF, URL, plain text) into a
ChromaDB vector store, querying them with a retrieval augmented generation
chain, and managing collections.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import asyncio
import os

from .models import IngestURLRequest, IngestTextRequest, QueryRequest, IngestResponse
from .ingestion import DocumentIngester
from .retrieval import VectorStore, CHROMA_PERSIST_DIR
from .chain import RAGChain

load_dotenv()

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 50 * 1024 * 1024))  # 50 MB default
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()
]

vector_store = VectorStore()
ingester = DocumentIngester()
rag_chain = RAGChain(vector_store)

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    yield


app = FastAPI(title="RAG Q&A API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest/pdf", response_model=IngestResponse)
@limiter.limit("10/minute")
async def ingest_pdf(
    request: Request,
    file: UploadFile = File(...),
    collection_name: str = "default",
):
    _validate_collection_name(collection_name)

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_BYTES} byte limit")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File is not a valid PDF (bad magic bytes)")

    loop = asyncio.get_event_loop()
    try:
        chunks = await loop.run_in_executor(None, ingester.ingest_pdf, content, file.filename)
        added = await loop.run_in_executor(None, vector_store.add_documents, chunks, collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {e}")

    return IngestResponse(
        message=f"Successfully ingested {file.filename}",
        chunks_added=added,
        collection_name=collection_name,
    )


@app.post("/ingest/url", response_model=IngestResponse)
@limiter.limit("10/minute")
async def ingest_url(request: Request, body: IngestURLRequest):
    _validate_collection_name(body.collection_name)

    try:
        docs = await ingester.ingest_url(body.url)
        loop = asyncio.get_event_loop()
        added = await loop.run_in_executor(None, vector_store.add_documents, docs, body.collection_name)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"URL ingestion failed: {e}")

    return IngestResponse(
        message=f"Successfully ingested {body.url}",
        chunks_added=added,
        collection_name=body.collection_name,
    )


@app.post("/ingest/text", response_model=IngestResponse)
@limiter.limit("30/minute")
async def ingest_text(request: Request, body: IngestTextRequest):
    _validate_collection_name(body.collection_name)

    try:
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(None, ingester.ingest_text, body.text, body.source_name)
        added = await loop.run_in_executor(None, vector_store.add_documents, docs, body.collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text ingestion failed: {e}")

    return IngestResponse(
        message=f"Successfully ingested '{body.source_name}'",
        chunks_added=added,
        collection_name=body.collection_name,
    )


@app.post("/query")
@limiter.limit("20/minute")
async def query(request: Request, body: QueryRequest):
    _validate_collection_name(body.collection_name)
    return StreamingResponse(
        rag_chain.query_stream(body.question, body.collection_name, body.chat_history),
        media_type="text/event-stream",
    )


@app.get("/collections")
def list_collections():
    try:
        return {"collections": vector_store.list_collections()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {e}")


@app.delete("/collections/{collection_name}")
def delete_collection(collection_name: str):
    _validate_collection_name(collection_name)
    try:
        vector_store.delete_collection(collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {e}")
    return {"message": f"Deleted collection '{collection_name}'"}


def _validate_collection_name(name: str) -> None:
    import re
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name or ""):
        raise HTTPException(
            status_code=400,
            detail="collection_name must be 1-64 chars, alphanumeric / underscore / hyphen",
        )
