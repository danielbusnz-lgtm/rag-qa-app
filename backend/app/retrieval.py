"""Chroma vector store wrapper for document retrieval.

Manages named collections backed by a persistent Chroma database on disk.
Each collection uses OpenAI's ``text-embedding-3-small`` model for embeddings.
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import chromadb
import os

CHROMA_PERSIST_DIR = "./chroma_db"


class VectorStore:
    """A lazy cache over named Chroma collections.

    Collections are created on first access and reused for subsequent calls.
    All collections share a single ``PersistentClient`` and embedding model.

    Attributes:
        embeddings: The OpenAI embedding model used across all collections.
    """

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._stores: dict[str, Chroma] = {}

    def get_store(self, collection_name: str) -> Chroma:
        """Returns the Chroma store for a collection, creating it if needed.

        Args:
            collection_name: Name of the Chroma collection.
        """
        if collection_name not in self._stores:
            self._stores[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )

        return self._stores[collection_name]

    def add_documents(self, docs: list[Document], collection_name: str) -> int:
        """Inserts documents into a collection.

        Args:
            docs: Documents to embed and store.
            collection_name: Target collection.

        Returns:
            The number of documents added.
        """
        store = self.get_store(collection_name)
        store.add_documents(docs)
        return len(docs)

    def similarity_search(self, query: str, collection_name: str, k: int = 4) -> list[Document]:
        """Finds the closest documents to a query string.

        Args:
            query: The natural language search query.
            collection_name: Collection to search against.
            k: Number of results to return. Defaults to 4.

        Returns:
            The top ``k`` documents ranked by cosine similarity.
        """
        store = self.get_store(collection_name)
        return store.similarity_search(query, k=k)

    def list_collections(self) -> list[str]:
        """Returns the names of all existing collections."""
        return [c.name for c in self._client.list_collections()]

    def delete_collection(self, collection_name: str):
        """Removes a collection from Chroma and drops it from the local cache.

        Args:
            collection_name: Collection to delete.
        """
        store = self.get_store(collection_name)
        store.delete_collection()
        self._stores.pop(collection_name, None)


