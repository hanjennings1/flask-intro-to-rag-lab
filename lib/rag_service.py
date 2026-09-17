from __future__ import annotations

import re
from typing import Any

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "get",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "our",
    "should",
    "so",
    "the",
    "their",
    "to",
    "use",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "you",
    "your",
}


# --- TOKENIZATION -----------------------------------------------------------
def tokenize(text: str) -> set[str]:
    """Convert text into a set of searchable lowercase tokens.
    - Lowercase the text.
    - Extract word-like values.
    - Remove leading/trailing apostrophes.
    - Remove tokens with length <= 1.
    - Remove tokens in STOPWORDS.
    - Return a set of searchable terms.
    """
    lowered = text.lower()          # normalize case so matching is consistent
    raw_tokens = re.findall(r"[a-z0-9']+", lowered)  # pull out word-like chunks, drop punctuation

    tokens = set()
    for token in raw_tokens:
        cleaned = token.strip("'")  # remove stray apostrophes from the edges only
        if len(cleaned) <= 1:       # skip single-character leftovers
            continue
        if cleaned in STOPWORDS:    # skip common filler words
            continue
        tokens.add(cleaned)         # keep anything that survives both checks

    return tokens


# --- DOCUMENT TEXT ASSEMBLY -----------------------------------------------------------
def document_search_text(document: dict[str, Any]) -> str:
    """Combine searchable document fields into one text value.
    - Include title, category, tags, and text.
    """
    return " ".join(        # space-separated so words don't run together
        [
            document["title"],
            document["category"],
            " ".join(document["tags"]),  # flatten the tags list into one string first
            document["text"],
        ]
    )


# --- RELEVANCE SCORING -----------------------------------------------------------
def score_document(query: str, document: dict[str, Any]) -> dict[str, Any]:
    """Score a document using keyword overlap.
        - Tokenize the query.
        - Tokenize the combined searchable document text.
        - Tokenize the document title.
        - Find matched terms between query tokens and document tokens.
        - Add a small title boost: 0.5 for each query token found in the title.
        - Return a dictionary with keys: document, score, matched_terms.
        """
    query_tokens = tokenize(query)                  # break the user's question into searchable terms

    document_text = document_search_text(document)  # combine title/category/tags/text
    document_tokens = tokenize(document_text)       # tokenize that combined blob

    title_tokens = tokenize(document["title"])      # tokenize just the title, for the boost

    matched_terms = query_tokens & document_tokens  # set intersection: terms in both

    base_score = len(matched_terms)                 # 1 point per matched term
    title_boost = sum(0.5 for token in query_tokens if token in title_tokens)  # extra credit for title matches
    score = base_score + title_boost

    return {
        "document": document,
        "score": score,
        "matched_terms": list(matched_terms),       # convert to list for JSON/iteration friendliness,
    }


# --- RETRIEVAL -----------------------------------------------------------
def retrieve_context(
    query: str,
    documents: list[dict[str, Any]],
    limit: int = 2,
    minimum_score: float = 1.0,
) -> list[dict[str, Any]]:
    """Select the most relevant documents for the query.
        - Score all documents.
        - Keep only matches with score >= minimum_score.
        - Sort by score from highest to lowest.
        - Return only the top `limit` matches.
    The selected context must depend on the user's query. Do not return the same
    hardcoded document for every request.
    """
    scored = [score_document(query, document) for document in documents]        # score every document

    relevant = [match for match in scored if match["score"] >= minimum_score]   # drop weak matches

    relevant.sort(key=lambda match: match["score"], reverse=True)               # highest score first

    return relevant[:limit]  # keep only the top N


# --- PROMPT CONSTRUCTION: FORMAT DOCS -----------------------------------------------------------
def format_context(context_matches: list[dict[str, Any]]) -> str:
    """Format retrieved documents into a context block for the prompt.
    - If no matches exist, return a short no-context message.
    - For each match, include Source ID, Title, Category, and Content.
    - Separate document blocks clearly.
    """
    if not context_matches:             # nothing scored high enough
        return "No relevant context was found for this query."

    blocks = []
    for match in context_matches:
        document = match["document"]    # pull the actual document out of the match dict
        block = (
            f"Source ID: {document['id']}\n"
            f"Title: {document['title']}\n"
            f"Category: {document['category']}\n"
            f"Content: {document['text']}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)  # blank lines + a divider between each document

# --- PROMPT CONSTRUCTION: BUILD PROMPT -----------------------------------------------------------
def build_prompt(query: str, context_matches: list[dict[str, Any]]) -> str:
    """Build a structured prompt with instructions, context, question, and requirements.
    The prompt should include these sections:
    - Instructions
    - Context
    - Question
    - Response requirements
    The prompt should tell the model to use only the provided context and avoid
    inventing unsupported details.
    """
    context_block = format_context(context_matches)  # reuse the formatter already built

    instructions = (
        "You are an internal assistant. Use only the information provided in the "
        "Context section below to answer the question. Do not invent unsupported "
        "details or rely on outside knowledge."
    )

    requirements = (
        "Answer clearly and concisely. Base your answer only on the provided "
        "context. If the context does not contain enough information, say so."
    )

    return (
        f"Instructions:\n{instructions}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{query}\n\n"
        f"Response requirements:\n{requirements}"
    )


# --- RESPONSE FORMATTING -----------------------------------------------------------
def source_metadata(match: dict[str, Any]) -> dict[str, str]:
    """Return source information that is safe to expose in the API response.
    Return only the document id and title.
    """
    document = match["document"]    # pull the full document out of the match

    return {
        "id": document["id"],       # safe to expose
        "title": document["title"], # safe to expose
    }

    # everything else on `document` (text, tags, category) and on `match` (score, matched_terms) 
    # is intentionally left out of the response