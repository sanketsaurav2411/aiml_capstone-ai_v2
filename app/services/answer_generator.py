from __future__ import annotations
from typing import List, Dict, Any

import os
from typing import Any

import requests

class AnswerGenerator:
    """Generate a relevant answer from retrieved documents."""

    def __init__(self, gemini_api_key: str | None = None) -> None:
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

    def generate(self, query: str, documents: list[dict[str, Any]]) -> str:
        if self.gemini_api_key:
            return self._generate_with_gemini(query, documents)
        return self._generate_locally(query, documents)

    def _generate_locally(self, query: str, documents: list[dict[str, Any]]) -> str:
        if not documents:
            return "I could not find relevant information for that question in the vector database."

        relevant_snippets = []
        for document in documents[:5]:
            content = document.get("content") or ""
            if content:
                relevant_snippets.append(content.strip())

        context = "\n\n".join(relevant_snippets)
        if not context:
            return "I found matching records, but their content is empty or unavailable."

        answer = (
            "Based on the retrieved context, here is a likely answer:\n\n"
            f"{context[:3000]}"
            "\n\nIf you need a more precise answer, refine your query or provide more context."
        )
        return answer

    def _generate_with_gemini(self, query: str, documents: list[dict[str, Any]]) -> str:
        prompt = "You are a helpful assistant. Answer the user's question using only the provided context."
        context = "\n\n".join(
            (document.get("content") or "").strip() for document in documents if document.get("content")
        )

        if not context:
            return self._generate_locally(query, documents)

        payload = {
            "system_instruction": {"parts": [{"text": prompt}]},
            "contents": [{"parts": [{"text": f"Question: {query}\n\nContext:\n{context}"}]}],
            "generationConfig": {"temperature": 0.2},
        }

        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent",
            params={"key": self.gemini_api_key},
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        payload_data = response.json()
        return payload_data["candidates"][0]["content"]["parts"][0]["text"].strip()
