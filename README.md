# Lab: Context-Aware RAG Endpoint with Flask
**Completed Sept 17, 2026**

## Overview

A simplified Retrieval-Augmented Generation (RAG) API built with Flask. The endpoint accepts an employee or developer question, retrieves relevant context from a set of approved internal company documents, builds a structured prompt, sends it to a local AI model (Ollama running `llama3.2`), and returns a generated answer along with the source documents that backed it.

This project simulates an internal assistant for a company's platform team, one that answers common employee questions (travel reimbursement, parental leave, API authentication, security incident reporting, software access, data retention) using only approved documentation, rather than letting a standalone model guess or hallucinate.

## How It Works

The request flow follows a straightforward RAG pipeline:

```
User question
   ↓
Tokenize the query
   ↓
Score every company document by keyword overlap (title, category, tags, and body text)
   ↓
Keep only documents above a minimum relevance score, sorted highest first
   ↓
Format the top matches into a context block
   ↓
Build a structured prompt (Instructions / Context / Question / Response requirements)
   ↓
Send the prompt to the AI model
   ↓
Return the generated answer + source IDs and titles
```

If no document scores highly enough for a given query, the model is never called, *instead* the API returns a safe fallback message immediately with an empty `sources` list. If the AI model itself fails (for example, Ollama isn't running), the API returns a `503` with a helpful error message instead of crashing.

## Project Structure

```
lib/
├── app.py                    # Flask app and the /api/ask route
├── rag_service.py            # Tokenizing, scoring, retrieval, prompt construction, source metadata
├── ai_client.py              # Sends prompts to Ollama and handles model-service errors
├── company_documents.py      # The approved internal document dataset
└── tests/
    ├── test_app.py           # Tests for the Flask route
    └── test_rag_service.py   # Tests for the RAG service functions
Pipfile
pytest.ini
```

## Setup

Install dependencies and activate the virtual environment:

```bash
pipenv install
pipenv shell
```

Run the test suite:

```bash
pytest
```

Run the Flask app manually:

```bash
cd lib
flask --app app run --debug
```

Check the health route (in a separate terminal):

```bash
curl -i http://127.0.0.1:5000/api/health
```

Expected response:

```json
{
  "status": "ok"
}
```

## The `/api/ask` Endpoint

**Request**

```
POST /api/ask
Content-Type: application/json

{
  "query": "How do I request software access?"
}
```

**Successful, context-backed response — `200`**

```json
{
  "query": "How do I request software access?",
  "answer": "...generated answer...",
  "sources": [
    {
      "id": "ops_software_access",
      "title": "Software Access Request Process"
    }
  ]
}
```

**Missing or blank query — `400`**

```json
{
  "error": "A non-empty 'query' string is required."
}
```

**Query with no matching documentation — `200`**

```json
{
  "query": "What is served in the cafeteria today?",
  "answer": "The approved company documents do not contain enough information to answer that question.",
  "sources": []
}
```

**Model-service failure — `503`**

```json
{
  "error": "Could not connect to Ollama. Make sure Ollama is installed, running, and that the llama3.2 model has been pulled."
}
```

Manual testing with Ollama running locally (`ollama pull llama3.2` first):

```bash
curl -i -X POST http://127.0.0.1:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I submit receipts for travel reimbursement?"}'
```

## Retrieval Approach

Retrieval is deliberately simple and keyword-based — no embeddings, semantic search, or vector database:

- **`tokenize()`** lowercases text, extracts word-like tokens, strips stray apostrophes, and drops both single-character tokens and common stopwords.
- **`document_search_text()`** combines each document's title, category, tags, and body text into one searchable string.
- **`score_document()`** compares tokenized query terms against tokenized document terms, counting overlapping ("matched") terms as the base score, with a small `+0.5` boost per query term that also appears in the document's title.
- **`retrieve_context()`** scores every document, filters out anything below a minimum score threshold, sorts by score (highest first), and returns only the top matches.

This keeps the retrieval logic transparent and easy to reason about, while still producing genuinely query-dependent results — different questions surface different source documents.

## Testing

All required behavior is covered by the provided pytest suite:

- Tokenizing and searchable-text assembly
- Document scoring and query-dependent retrieval
- Context formatting, prompt construction, and source metadata shaping
- Request validation (missing, blank, and non-string queries)
- Full RAG workflow integration through the Flask route
- Safe fallback behavior when no context is found
- Graceful `503` handling when the model service fails

Run everything with:

```bash
pytest -v
```

## Reflection

This lab was built incrementally, from the smallest function outward: `tokenize()` first, then each subsequent piece layered on top and verified with pytest before moving on. That order made regressions easy to catch early; for example, a missing separator when joining document fields would have silently broken keyword matching if it hadn't been tested immediately.

The scoring logic (`score_document()`) was the most involved part, relying on set intersection to find matched terms and a small title-based boost to bias results toward closely-matching documents. The Flask route itself stays thin by design. It composes the already-tested `rag_service.py` functions in sequence and handles two edge cases (no context found, model-service failure) as early exits, rather than duplicating any retrieval or formatting logic inline.

