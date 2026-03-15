# rag-qa-app

A document Q&A app built with RAG (Retrieval-Augmented Generation). Upload PDFs, paste URLs, or add plain text — then ask questions against your documents with streaming answers and source citations.

Live at [rag-qa-api.com](https://rag-qa-api.com)

## Stack

- **FastAPI** — REST API with streaming responses
- **LangChain** — document loading and RAG pipeline
- **ChromaDB** — local vector store
- **OpenAI** — text-embedding-3-small for embeddings, gpt-4o-mini for answers
- **React** — frontend UI

## Setup

```bash
git clone https://github.com/danielbusnz-lgtm/rag-qa-app
cd rag-qa-app
```

**Backend**

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r backend/requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-key-here
```

```bash
uvicorn backend.app.main:app --reload
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest/pdf` | Upload a PDF |
| POST | `/ingest/url` | Ingest a URL |
| POST | `/ingest/text` | Ingest plain text |
| POST | `/query` | Ask a question (streaming) |
| GET | `/collections` | List collections |
| DELETE | `/collections/{name}` | Delete a collection |

## Running with Docker

```bash
docker-compose up --build
```

## Tests

```bash
pytest
```

## Features

- Ingest PDFs, URLs, and plain text
- Streaming answers via Server-Sent Events
- Source citations on every response
- Conversation history
- Multiple named document collections
