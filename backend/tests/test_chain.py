import pytest
import asyncio
from unittest.mock import MagicMock
from backend.app.chain import RAGChain
from backend.app.retrieval import VectorStore
from langchain.schema import Document


async def empty_stream(*args, **kwargs):
    return 
    yield
    
@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=VectorStore)
    store.similarity_search.return_value = [
        Document(
            page_content="Guido van Rossum created Python in 1991.",
            metadata={"source": "test.pdf", "page": 1}
        )
    ]
    return store

@pytest.fixture
def rag_chain(mock_vector_store):
    chain = RAGChain(mock_vector_store)
    chain.llm = MagicMock()
    chain.llm.astream = empty_stream
    return chain

def test_rag_chain_searches_correct_collection(rag_chain, mock_vector_store):
    async def run():
        async for _ in rag_chain.query_stream("who created Python?", "study-docs"):
            pass
    asyncio.run(run())
    mock_vector_store.similarity_search.assert_called_once_with("who created Python?", "study-docs")

def test_rag_chain_uses_chat_history(rag_chain, mock_vector_store):
    history = [("What is Python?", "Python is a programming language.")]
    async def run():
        async for _ in rag_chain.query_stream("who created it?", "study-docs", history):
            pass
    asyncio.run(run())
    mock_vector_store.similarity_search.assert_called_once_with("who created it?", "study-docs")
