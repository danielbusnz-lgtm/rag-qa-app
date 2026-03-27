"""Pydantic request and response models for the RAG Q&A API.

These models define the shape of data flowing through the ingestion
and query endpoints.
"""

from pydantic import BaseModel
from typing import Optional


class IngestURLRequest(BaseModel):
    """A request to ingest a document from a URL.

    Attributes:
        url: The URL to fetch and ingest.
        collection_name: Target collection. Defaults to ``"default"``.
    """

    url: str
    collection_name: str = "default"


class IngestTextRequest(BaseModel):
    """A request to ingest raw text directly.

    Attributes:
        text: The text content to ingest.
        source_name: Label for the source of this text.
        collection_name: Target collection. Defaults to ``"default"``.
    """

    text: str
    source_name: str = "manual_input"
    collection_name: str = "default"


class QueryRequest(BaseModel):
    """A question posed against a document collection.

    Attributes:
        question: The natural language question to answer.
        collection_name: Which collection to search. Defaults to ``"default"``.
        chat_history: Prior turns as ``(human, ai)`` pairs, used for
            conversational context. Empty by default.
    """

    question: str
    collection_name: str = "default"
    chat_history: list[tuple[str, str]] = []


class IngestResponse(BaseModel):
    """Result of an ingestion request.

    Attributes:
        message: Human readable status message.
        chunks_added: Number of text chunks stored.
        collection_name: The collection that received the chunks.
    """

    message: str
    chunks_added: int
    collection_name: str


class DocumentSource(BaseModel):
    """A single source document returned alongside a query answer.

    Attributes:
        source: Where the chunk came from (URL, filename, etc.).
        page: Page number within the source, if applicable.
        preview: Short excerpt of the matching text.
    """

    source: str
    page: Optional[int] = None
    preview: str


