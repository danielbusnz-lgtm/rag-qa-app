"""Pydantic request and response models for the RAG Q&A API.

These models define the shape of data flowing through the ingestion
and query endpoints.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re

COLLECTION_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_collection(v: str) -> str:
    if not COLLECTION_RE.fullmatch(v or ""):
        raise ValueError("collection_name must be 1-64 chars, alphanumeric / underscore / hyphen")
    return v


class IngestURLRequest(BaseModel):
    """A request to ingest a document from a URL.

    Attributes:
        url: The URL to fetch and ingest.
        collection_name: Target collection. Defaults to ``"default"``.
    """

    url: str
    collection_name: str = "default"

    _v_collection = field_validator("collection_name")(_validate_collection)


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

    _v_collection = field_validator("collection_name")(_validate_collection)


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

    _v_collection = field_validator("collection_name")(_validate_collection)


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


