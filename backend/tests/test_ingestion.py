import pytest
from backend.app.ingestion import DocumentIngester


@pytest.fixture
def ingester():
    return DocumentIngester()


def test_ingest_text_returns_documents(ingester):
    docs = ingester.ingest_text("Python was created by Guido van Rossum.", "test-source")
    assert len(docs) > 0


def test_ingest_text_preserves_source(ingester):
    docs = ingester.ingest_text("Python was created by Guido van Rossum.", "test-source")
    assert all(d.metadata["source"] == "test-source" for d in docs)


def test_ingest_text_chunks_large_text(ingester):
    large_text = "Python is great. " * 500
    docs = ingester.ingest_text(large_text, "test-source")
    assert len(docs) > 1


def test_ingest_text_chunk_size_respected(ingester):
    large_text = "word " * 2000
    docs = ingester.ingest_text(large_text, "test-source")
    # Chunks should be bounded by CHUNK_SIZE (default 1000)
    assert all(len(d.page_content) <= 1200 for d in docs)
