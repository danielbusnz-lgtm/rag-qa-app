from langchain_community.document_loaders import PyMuPDFLoader, AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tempfile, os


class DocumentIngester:
    def __init__(self):
        self.text_splitter =  RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,

        )
    def ingest_pdf(self, file_bytes: bytes, filename: str) -> list[Document]:
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
            os.unlink(tmp_path)

    async def ingest_url(self, url:str) -> list[Document]:
        loader = AsyncHtmlLoader([url])
        docs = await loader.aload()
        transformer = Html2TextTransformer()
        docs = list(transformer.transform_documents(docs))
        for doc in docs:
            doc.metadata["source"] = url
        return self.text_splitter.split_documents(docs)

    def ingest_text(self, text: str, source_name: str) -> list[Document]:
        doc = Document(page_content=text, metadata={"source": source_name})
        return self.text_splitter.split_documents([doc])
