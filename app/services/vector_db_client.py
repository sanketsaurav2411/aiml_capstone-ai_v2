from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import requests

try:
    import chromadb
except ImportError:  # pragma: no cover - optional dependency for Chroma-backed vector stores
    chromadb = None

from app.config import DEFAULT_COLLECTION, VECTOR_DB_BASE_URL
from app.core.errors import VectorDBConnectionError


class VectorDBClient:
    """Small client for querying a local vector database over HTTP."""

    def __init__(
        self,
        base_url: str = VECTOR_DB_BASE_URL,
        default_collection: str | None = DEFAULT_COLLECTION,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_collection = default_collection
        self.timeout = timeout
        self._chroma_client = self._create_chroma_client()

    def _create_chroma_client(self) -> Any | None:
        if chromadb is None:
            return None

        try:
            parsed = urlparse(self.base_url)
            return chromadb.HttpClient(
                host=parsed.hostname or "localhost",
                port=parsed.port or (443 if parsed.scheme == "https" else 8000),
                ssl=parsed.scheme == "https",
                tenant="default_tenant",
                database="default_database",
            )
        except Exception:  # pragma: no cover - connection is validated in real integration
            return None

    def search(
        self,
        query: str,
        collection: str | None = None,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        target_collection = collection or self.default_collection

        if self._chroma_client is not None and target_collection:
            try:
                collection_handle = self._chroma_client.get_collection(target_collection)
                result = collection_handle.query(
                    query_texts=[query],
                    n_results=min(top_k, 10),
                    include=["documents", "metadatas", "distances"],
                    where=filters,
                )
                return self._normalize_chroma_response(result)
            except Exception as exc:  # pragma: no cover - exercised via real Chroma server integration
                last_error = exc
            else:
                return []
        else:
            last_error = None

        payload: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "filter": filters or {},
        }
        if target_collection:
            payload["collection"] = target_collection

        for endpoint in self._candidate_endpoints(target_collection):
            try:
                response = requests.post(endpoint, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:  # pragma: no cover - exercised via live integration
                last_error = exc
                continue

            if response.status_code in {404, 405}:
                last_error = RuntimeError(f"Endpoint not found: {endpoint}")
                continue

            if response.status_code >= 400:
                last_error = RuntimeError(
                    f"Vector DB request failed for {endpoint}: {response.status_code} {response.text}"
                )
                continue

            try:
                data = response.json()
            except ValueError as exc:
                last_error = exc
                continue

            normalized_documents = self._normalize_response(data)
            if normalized_documents:
                return normalized_documents

            return []

        raise VectorDBConnectionError(
            "Unable to reach the vector database. "
            f"Checked endpoints: {', '.join(self._candidate_endpoints(target_collection))}. "
            f"Last error: {last_error}"
        )

    def _candidate_endpoints(self, collection: str | None) -> list[str]:
        endpoints: list[str] = [
            f"{self.base_url}/query",
            f"{self.base_url}/search",
            f"{self.base_url}/api/v1/query",
            f"{self.base_url}/api/v1/search",
        ]

        if collection:
            endpoints = [
                f"{self.base_url}/collections/{collection}/query",
                f"{self.base_url}/collections/{collection}/search",
                f"{self.base_url}/{collection}/query",
                f"{self.base_url}/{collection}/search",
                *endpoints,
            ]

        return endpoints

    def _normalize_chroma_response(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        distances = data.get("distances") or []
        ids = data.get("ids") or []

        normalized: list[dict[str, Any]] = []
        for batch_index, batch_documents in enumerate(documents):
            batch_metadata = metadatas[batch_index] if batch_index < len(metadatas) else []
            batch_distances = distances[batch_index] if batch_index < len(distances) else []
            batch_ids = ids[batch_index] if batch_index < len(ids) else []

            for item_index, content in enumerate(batch_documents):
                metadata = batch_metadata[item_index] if item_index < len(batch_metadata) else {}
                doc_id = batch_ids[item_index] if item_index < len(batch_ids) else None
                distance = batch_distances[item_index] if item_index < len(batch_distances) else None
                normalized.append(
                    {
                        "id": doc_id,
                        "content": content,
                        "score": None if distance is None else max(0.0, 1.0 - float(distance)),
                        "metadata": metadata,
                    }
                )

        return normalized

    def _normalize_response(self, data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for key in ("results", "matches", "documents", "data", "items"):
                if key in data:
                    items = data[key]
                    break
            else:
                return []
        else:
            return []

        normalized: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, str):
                normalized.append({"content": item})
                continue

            if not isinstance(item, dict):
                continue

            content = (
                item.get("content")
                or item.get("text")
                or item.get("page_content")
                or item.get("chunk")
            )
            metadata = item.get("metadata") or {}
            doc_payload = item.get("document")
            if isinstance(doc_payload, dict):
                content = content or doc_payload.get("content") or doc_payload.get("text")
                metadata = metadata or doc_payload.get("metadata") or {}

            if isinstance(item.get("embedding"), list):
                metadata["embedding_length"] = len(item["embedding"])

            normalized.append(
                {
                    "id": item.get("id") or item.get("doc_id") or item.get("uuid"),
                    "content": str(content) if content is not None else "",
                    "score": item.get("score"),
                    "metadata": metadata,
                }
            )

        return normalized
