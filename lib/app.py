from __future__ import annotations

from flask import Flask, jsonify, request

from lib.ai_client import generate_response
from lib.company_documents import COMPANY_DOCUMENTS
from lib.rag_service import build_prompt, retrieve_context, source_metadata


def create_app():
    app = Flask(__name__)

    @app.get("/api/health")
    def health_check():
        return jsonify({"status": "ok"})

    @app.post("/api/ask")
    def ask_question():
        """Accept a query and return a source-backed generated answer."""
        #. Read JSON request data safely:
        data = request.get_json(silent=True) or {}                      # avoid crashing on missing/invalid JSON body
        query = data.get("query")                                       # None if the key doesn't exist at all
        #. Validate that `query` is a non-empty string:
        if not isinstance(query, str) or not query.strip():
            # catches: missing key (None), non-string types (list/int/etc), and blank/whitespace strings
            return jsonify({"error": "A non-empty 'query' string is required."}), 400
        #. Retrieve relevant context from COMPANY_DOCUMENTS:
        context_matches = retrieve_context(query, COMPANY_DOCUMENTS)    # reuse the retrieval function
        #. If no context is found, return a safe fallback with an empty sources list.
        if not context_matches:                                         # no document scored high enough
            return jsonify({
                "query": query,
                "answer": "The approved company documents do not contain enough information to answer that question.",
                "sources": [],
            }), 200
        #. Build a structured prompt from the selected context:
        prompt = build_prompt(query, context_matches)                   # reuse the prompt builder
        #. Call generate_response(prompt):
        try:
            answer = generate_response(prompt)
        except RuntimeError as error:
            #. If generate_response raises RuntimeError, return a 503 service error:
            return jsonify({"error": str(error)}), 503 
        
        #. Return query, answer, and sources as JSON:    
        sources = [source_metadata(match) for match in context_matches]  # safe id/title only
        return jsonify({
            "query": query,
            "answer": answer,
            "sources": sources,
        }), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
