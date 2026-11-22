from __future__ import annotations

import os
from typing import Any

import requests

from app.config import GEMINI_API_KEY, GEMINI_MODEL


class AnswerGenerator:
    """Generate a relevant answer from retrieved documents."""

    def __init__(self, gemini_api_key: str | None = None, model: str | None = None) -> None:
        self.gemini_api_key = gemini_api_key or GEMINI_API_KEY or os.getenv("OPENAI_API_KEY")
        self.model = model or GEMINI_MODEL

    def generate(self, query: str, documents: list[dict[str, Any]]) -> str:
        if self.gemini_api_key:
            try:
                return self._generate_with_gemini(query, documents)
            except Exception:
                return self._generate_locally(query, documents)
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

        return (
            "Based on the retrieved Indian ITC-HS context, here is the relevant tariff evidence:\n\n"
            f"{context[:3000]}"
            "\n\nRefine your query or provide product specifications (material, function, end-use) for 8-digit classification."
        )

    def _generate_with_gemini(self, query: str, documents: list[dict[str, Any]]) -> str:
        prompt = (
            "You are an Indian Customs & ITC-HS 2022 Tariff Expert. "
            "Answer the user's question using ONLY the provided context and cite the HS code or note."
        )
        context = "\n\n".join(
            (document.get("content") or "").strip() for document in documents if document.get("content")
        )
        if not context:
            return self._generate_locally(query, documents)

        payload = {
            "system_instruction": {"parts": [{"text": prompt}]},
            "contents": [{"parts": [{"text": f"Question: {query}\n\nContext:\n{context}"}]}],
            "generationConfig": {"temperature": 0.1},
        }
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            params={"key": self.gemini_api_key},
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=25,
        )
        response.raise_for_status()
        payload_data = response.json()
        return payload_data["candidates"][0]["content"]["parts"][0]["text"].strip()
