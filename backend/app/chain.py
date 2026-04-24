"""RAG question answering chain with streaming output.

Wraps a ChatOpenAI model and a VectorStore to answer questions grounded
in retrieved documents. Responses stream back as Server-Sent Events (SSE)
so the frontend can render tokens incrementally.

Example::

    store = VectorStore()
    chain = RAGChain(store)
    async for event in chain.query_stream("What is X?", "my_docs"):
        print(event)
"""

from langchain_openai import ChatOpenAI
from langchain.schema import Document
from .retrieval import VectorStore
import json
import os

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, say "I don't have enough information in the provided documents to answer this question."
Always cite your sources.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:"""


def _build_prompt(context: str, history_str: str, question: str) -> str:
    # Replace placeholders literally — avoids .format() breaking on user input
    # containing curly braces (prompt-injection vector).
    return (
        SYSTEM_PROMPT
        .replace("{context}", context)
        .replace("{chat_history}", history_str)
        .replace("{question}", question)
    )

class RAGChain:
    """A retrieval-augmented generation chain that streams answers over SSE.

    Attributes:
        vector_store: The backing store used for similarity search.
        llm: ChatOpenAI instance (gpt-4o-mini, temperature 0, streaming on).
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=0,
            streaming=True,
        )

    async def query_stream(self, question: str, collection_name: str, chat_history: list[tuple[str, str]] = []):
        """Stream an SSE answer for a question, grounded in retrieved docs.

        Retrieves the top matching documents from ``collection_name``, formats
        them alongside any prior conversation turns, and streams the LLM
        response token by token. After the last token, a ``sources`` event
        is emitted with metadata for each retrieved document, followed by a
        ``[DONE]`` sentinel.

        Args:
            question: The user's question in plain text.
            collection_name: Which Chroma collection to search against.
            chat_history: Prior (human, assistant) exchange pairs. Defaults
                to an empty list.

        Yields:
            SSE formatted strings. Each is one of three kinds: a ``token``
            event carrying a chunk of the answer, a ``sources`` event with
            document metadata, or the final ``[DONE]`` marker.
        """
        try:
            docs = self.vector_store.similarity_search(question, collection_name)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': f'Retrieval failed: {e}'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
            for doc in docs
        ])

        history_str = "\n".join([
            f"Human: {h}\nAssistant: {a}"
            for h, a in chat_history
        ])

        prompt = _build_prompt(context, history_str, question)
        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "preview": doc.page_content[:200],
            }
            for doc in docs
        ]

        try:
            async for chunk in self.llm.astream(prompt):
                yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': f'LLM stream failed: {e}'})}\n\n"

        yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"
        yield "data: [DONE]\n\n"



