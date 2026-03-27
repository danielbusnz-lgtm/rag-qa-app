"""FastAPI backend for a RAG Q&A application.

Exposes endpoints for ingesting documents (PDF, URL, plain text) into a
ChromaDB vector store, querying them with a retrieval augmented generation
chain, and managing collections.

Example:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import asyncio
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
    """Ensure the ChromaDB storage directory exists before the app starts."""
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
    """Return a simple status check."""
    return {"status": "ok"}

@app.post("/ingest/pdf", response_model = IngestResponse)
async def ingest_pdf(
        file: UploadFile = File(...),
        collection_name: str = "default"
        ):
    """Parse a PDF upload, chunk it, and store the vectors.

    Args:
        file: The uploaded PDF. Rejects non PDF files with a 400.
        collection_name: Target collection in the vector store.
            Defaults to ``"default"``.

    Returns:
        An IngestResponse with the filename, chunk count, and collection.

    Raises:
        HTTPException: If the file is not a PDF.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail = "only pdf files supported")
    content = await file.read()
    loop = asyncio.get_event_loop()
    chunks = await loop.run_in_executor(None, ingester.ingest_pdf, content, file.filename)
    added = await loop.run_in_executor(None, vector_store.add_documents, chunks, collection_name)
    return IngestResponse(
        message=f"Successfully ingested {file.filename}",
        chunks_added =added,
        collection_name = collection_name
    )

@app.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(request: IngestURLRequest):
    """Scrape a URL, chunk its content, and add the vectors to a collection.

    Args:
        request: Contains the target URL and optional collection name.

    Returns:
        An IngestResponse confirming how many chunks were stored.
    """
    docs = await ingester.ingest_url(request.url)
    loop = asyncio.get_event_loop()
    added = await loop.run_in_executor(None, vector_store.add_documents, docs, request.collection_name)
    return IngestResponse(
        message=f"Successfully ingested {request.url}",
        chunks_added=added,
        collection_name=request.collection_name
    )

@app.post("/ingest/text", response_model = IngestResponse)
async def ingest_text(request: IngestTextRequest):
    """Chunk raw text and store the resulting vectors.

    Args:
        request: Contains the text body, a source name label, and an
            optional collection name.

    Returns:
        An IngestResponse with the chunk count and collection.
    """
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(None, ingester.ingest_text, request.text, request.source_name)
    added = await loop.run_in_executor(None, vector_store.add_documents, docs, request.collection_name)
    return IngestResponse(
        message = f"Successfully ingested '{request.source_name}'",
        chunks_added = added,
        collection_name = request.collection_name
    )

@app.post("/query")
async def query(request: QueryRequest):
    """Run a RAG query and stream the answer back as server sent events.

    Retrieves relevant chunks from the vector store, feeds them into the
    LLM chain along with any prior chat history, and streams tokens as
    they are generated.

    Args:
        request: The question, target collection, and optional chat history.

    Returns:
        A streaming ``text/event-stream`` response.
    """
    return StreamingResponse(
        rag_chain.query_stream(
            request.question,
            request.collection_name,
            request.chat_history

        ),
        media_type="text/event-stream"
    )

@app.get("/collections")
def list_collections():
    """Return the names of all collections in the vector store."""
    return {"collections": vector_store.list_collections()}

@app.delete("/collections/{collection_name}")
def delete_collection(collection_name: str):
    """Drop a collection and all its stored vectors.

    Args:
        collection_name: The collection to delete.
    """
    vector_store.delete_collection(collection_name)
    return {"message": f"Deleted collection '{collection_name}'"}
