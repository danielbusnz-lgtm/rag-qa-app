from pydantic import BaseModel
from typing import Optional

class IngestURLRequest(BaseModel):
    url: str
    collection_name: str = "default"

class IngestTextRequest(BaseModel):
    text: str
    source_name: str = "manual_input"
    collection_name: str = "default"

class QueryRequest(BaseModel):
    question: str
    collection_name: str = "default"
    chat_history: list[tuple[str, str]] = []

class IngestResponse(BaseModel):
    message: str
    chunks_added: int
    collection_name: str

class DocumentSource(BaseModel):
    source: str
    page: Optional[int] = None
    preview: str


