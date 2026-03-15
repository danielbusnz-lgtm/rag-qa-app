from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import os

CHROMA_PERSIST_DIR = "./chroma_db"

class VectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._stores: dict[str, Chroma] = {}
    def get_store(self, collection_name: str) -> Chroma:
        if collection_name not in self._stores:
            self._stores[collection_name] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )

        return self._stores[collection_name]
    
    def add_documents(self, docs:list[Document], collection_name:str) -> int:
        store = self.get_store(collection_name)
        store.add_documents(docs)
        return len(docs)
    
    def similarity_search(self, query:str, collection_name: str, k: int = 4) -> list[Document]:
        store = self.get_store(collection_name)
        return store.similarity_search(query, k=k)
    
    def list_collections(self) -> list[str]:
        if not os.path.exists(CHROMA_PERSIST_DIR):
            return []
        return [d for d in os.listdir(CHROMA_PERSIST_DIR)
                if os.path.isdir(os.path.join(CHROMA_PERSIST_DIR, d))]
    
    def delete_collection(self, collection_name: str):
        store = self.get_store(collection_name)
        store.delete_collection()
        self._stores.pop(collection_name, None)


