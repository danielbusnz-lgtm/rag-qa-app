"""Document ingestion pipeline for the RAG QA app.

Handles loading documents from PDFs, URLs, and raw text, then splits them
into chunks suitable for embedding and retrieval.

Example::

    ingester = DocumentIngester()
    chunks = ingester.ingest_text("Some long article...", source_name="notes.txt")
"""

from langchain_community.document_loaders import PyMuPDFLoader, AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import asyncio
import contextlib
import tempfile
import os

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
URL_FETCH_TIMEOUT_S = int(os.getenv("URL_FETCH_TIMEOUT_S", 15))


class DocumentIngester:
    """A document ingestion pipeline that loads and chunks content for RAG.

    Supports three input types: PDF files (as raw bytes), web URLs, and
    plain text strings. Each method returns chunked LangChain Documents
    with source metadata attached.

    Attributes:
        text_splitter: Splits documents into overlapping 1000 character
            chunks with 200 character overlap.
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    def ingest_pdf(self, file_bytes: bytes, filename: str) -> list[Document]:
        """Load a PDF from bytes, tag it with a source name, and chunk it.

        Writes the bytes to a temporary file for PyMuPDF to read, then
        cleans up the temp file regardless of success or failure.

        Args:
            file_bytes: Raw PDF content.
            filename: Human readable name stored in each chunk's
                ``metadata["source"]``.

        Returns:
            Chunked documents with source metadata set to ``filename``.
        """
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(file_bytes)
            tmp_path = f.name
        try:
            loader = PyMuPDFLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            return self.text_splitter.split_documents(docs)
        finally:
            with contextlib.suppress(FileNotFoundError, OSError):
                os.unlink(tmp_path)

    async def ingest_url(self, url: str) -> list[Document]:
        loader = AsyncHtmlLoader([url])
        docs = await asyncio.wait_for(loader.aload(), timeout=URL_FETCH_TIMEOUT_S)
        transformer = Html2TextTransformer()
        docs = list(transformer.transform_documents(docs))
        for doc in docs:
            doc.metadata["source"] = url
        return self.text_splitter.split_documents(docs)

    def ingest_text(self, text: str, source_name: str) -> list[Document]:
        """Wrap raw text in a Document and chunk it.

        Args:
            text: The content to ingest.
            source_name: Label stored in each chunk's ``metadata["source"]``.

        Returns:
            Chunked documents ready for embedding.
        """
        doc = Document(page_content=text, metadata={"source": source_name})
        return self.text_splitter.split_documents([doc])
