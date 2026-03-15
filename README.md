# rag-qa-app

A document question-answering system built with LangChain, ChromaDB, and OpenAI. Upload PDFs, web pages, or plain text and ask questions against them.

## Stack

- **FastAPI** — REST API
- **LangChain** — document loading and RAG pipeline
- **ChromaDB** — local vector database
- **OpenAI** — embeddings and completions
- **Streamlit** — frontend UI

## Getting started

```bash
git clone https://github.com/danielbusnz-lgtm/rag-qa-app
cd rag-qa-app
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt
```

Copy `.env.example` to `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your-key-here
```

## Running the app

```bash
uvicorn backend.app.main:app --reload
```

## Features

- Ingest PDFs, URLs, and plain text
- Ask questions with streaming responses
- Source citations on every answer
- Conversation history
- Multiple document collections
