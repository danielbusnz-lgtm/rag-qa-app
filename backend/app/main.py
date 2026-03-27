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
    return {"collections": vector_store.list_collections()}

@app.delete("/collections/{collection_name}")
def delete_collection(collection_name: str):
    vector_store.delete_collection(collection_name)
    return {"message": f"Deleted collection '{collection_name}'"}
