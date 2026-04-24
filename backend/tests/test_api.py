"""Endpoint-level tests using FastAPI TestClient."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    # Stub OpenAIEmbeddings so import doesn't hit the network
    with patch("backend.app.retrieval.OpenAIEmbeddings"):
        from backend.app import main  # import after env is set
        with TestClient(main.app) as c:
            yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_pdf_rejects_bad_magic_bytes(client):
    r = client.post(
        "/ingest/pdf",
        files={"file": ("evil.pdf", b"not a real pdf", "application/pdf")},
    )
    assert r.status_code == 400
    assert "magic bytes" in r.json()["detail"].lower()


def test_pdf_rejects_oversized_upload(client, monkeypatch):
    # Lower the cap just for this test
    import backend.app.main as main_mod
    monkeypatch.setattr(main_mod, "MAX_UPLOAD_BYTES", 10)
    r = client.post(
        "/ingest/pdf",
        files={"file": ("big.pdf", b"%PDF" + b"x" * 100, "application/pdf")},
    )
    assert r.status_code == 413


def test_rejects_bad_collection_name(client):
    r = client.post("/ingest/text", json={
        "text": "hello",
        "source_name": "s",
        "collection_name": "bad name with spaces!",
    })
    assert r.status_code == 422  # pydantic validation

    r = client.delete("/collections/bad%20name")
    assert r.status_code == 400


def test_accepts_valid_collection_name(client):
    with patch("backend.app.main.vector_store") as vs:
        vs.add_documents.return_value = 1
        r = client.post("/ingest/text", json={
            "text": "hello world",
            "source_name": "s",
            "collection_name": "my-coll_1",
        })
    assert r.status_code == 200
