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
    assert docs[0].metadata["source"] == "test-source"

def test_ingest_text_chunks_large_text(ingester):
    large_text = "Python is great. " * 500
    docs = ingester.ingest_text(large_text, "test-source")
    assert len(docs) > 1
