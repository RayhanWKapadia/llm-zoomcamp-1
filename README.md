# LLM Zoomcamp - RAG Project

A Retrieval-Augmented Generation (RAG) system built as part of the [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) course. The project progresses from a basic FAQ-focused RAG to a generalized, domain-agnostic RAG with both synchronous and agentic query modes.

## Overview

The system retrieves relevant documents from a search index and feeds them as context to an LLM (OpenAI) to answer questions accurately. It supports:

- **Synchronous RAG** — single search → build prompt → call LLM
- **Agentic RAG** — LLM iteratively calls a search tool multiple times before producing a final answer

## Project Structure

```
.
├── rag_helper.py              # Base RAG class (course-specific, LLM Zoomcamp FAQ)
├── agentic_rag_helper.py      # Agentic RAG class using toyaikit tool-calling loop
├── generalized_rag.py         # Domain-agnostic RAG (sync + agentic, configurable)
├── main_rag.ipynb             # Main notebook: building and running the RAG pipeline
├── medical_rag.ipynb          # RAG applied to a medical knowledge domain
├── sqllite-ingest.ipynb       # Ingesting FAQ data into SQLite via sqlitesearch
├── notebook.ipynb             # Scratch / exploration notebook
└── helperfiles/
    ├── ingest.py              # Fetches FAQ data from DataTalks.Club and builds minsearch index
    ├── rag_helper.py          # Earlier version of the RAG base class
    └── db/
        └── faq.db             # SQLite database with ingested FAQ data
```

## Key Classes

### `RAGBase` ([rag_helper.py](rag_helper.py))
Basic RAG pipeline scoped to the LLM Zoomcamp FAQ. Searches with field boosts on `question` and `section`, formats context as Q&A blocks, and calls the OpenAI Responses API.

### `AgenticRAGBase` ([agentic_rag_helper.py](agentic_rag_helper.py))
Extends the synchronous approach with a `toyaikit` agentic loop. The LLM is given a `agentic_search` tool and can call it multiple times with different queries before returning an answer.

### `GeneralizedRAG` ([generalized_rag.py](generalized_rag.py))
Domain-agnostic drop-in. Configure `domain`, `boost_dict`, `filter_dict`, `context_fields`, and the system prompt is auto-generated. Supports both `.rag()` (synchronous) and `.agenticRAG()` (tool-loop) modes.

```python
from generalized_rag import GeneralizedRAG
from openai import OpenAI

rag = GeneralizedRAG(
    index=fitted_index,
    llm_client=OpenAI(),
    domain="medical information",
    context_fields=["title", "content"],
    model="gpt-5.4-mini",
)

answer = rag.rag("What are the symptoms of anemia?")
answer = rag.agenticRAG("Explain treatment options for type 2 diabetes.")
```

## Setup

**Requirements**: Python 3.14+, [uv](https://github.com/astral-sh/uv)

```bash
uv sync
```

Set your OpenAI API key in a `.env` file:

```
OPENAI_API_KEY=sk-...
```

## Dependencies

| Package | Purpose |
|---|---|
| `openai` | LLM calls via Responses API |
| `minsearch` | In-memory full-text search index |
| `sqlitesearch` | SQLite-backed search index |
| `toyaikit` | Agentic tool-calling loop helpers |
| `gitsource` | Load documents from a Git repo |
| `requests` | Fetch FAQ data from DataTalks.Club |
| `python-dotenv` | Load `.env` credentials |
| `jupyter` | Notebook environment |
