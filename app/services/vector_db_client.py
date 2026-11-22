from __future__ import annotations

import logging
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

try:
    import chromadb
except ImportError:  # pragma: no cover - optional dependency
    chromadb = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - optional dependency
    BM25Okapi = None

from app.config import (
    CHROMA_PERSIST_DIR,
    DEFAULT_COLLECTION,
    EMBED_MODEL_NAME,
    ENABLE_LOCAL_EMBEDDINGS,
    VECTOR_DB_BASE_URL,
)
from app.core.errors import VectorDBConnectionError
from app.services.document_parser import filter_english_lines

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_EXCLUDED_RULING_FILES = {
    "NAC8_4_1775709082_1_af3b.pdf",
    "valuation_41324-41330_2025.pdf",
    "NAC10_7_533a0daa5738b0770306b2d78df27adc.pdf",
}

_RULING_OVERRIDES: dict[str, dict[str, str]] = {
    "3402 _ CAARMumARC1372025-26.pdf": {
        "hs_codes": "84561100",
        "authority": "CAAR",
        "order_ref": "CAAR/Mum/ARC/23/2026-27 (LaserPecker Laser Engraving Machine)",
    },
    "NAC10_2_a9acb7518dc01e1eb54824b7d688fcda.pdf": {
        "hs_codes": "34024200",
        "authority": "CAAR",
        "order_ref": "CAAR/Mum/ARC/137/2025-26 (Polysorbate 65 Surfactant)",
    },
    "7160 (as listed_ verify)_CAAR_Mum_ARC_22_2025-26.pdf": {
        "hs_codes": "76109010,76109020,84806000",
        "authority": "CAAR",
        "order_ref": "CAAR/Mum/ARC/22/2025-26 (Aluminium Formwork)",
    },
    "83069703_CAAR_CUS_APPL_77_2023 (registry reference).pdf": {
        "hs_codes": "97039010,83062110,74093900",
        "authority": "CAAR",
        "order_ref": "CAAR/Mum/ARC/61/2024 (Bronze Statue Panels)",
    },
    "8028090_CAAR_Del_Betelnut_109_2024.pdf": {
        "hs_codes": "08028090",
        "authority": "CAAR",
        "order_ref": "CAAR/Del/Betelnut/109/2024",
    },
}

_AUTHORITY_RANK = {
    "Supreme Court": 1,
    "High Court": 2,
    "CESTAT": 3,
    "CAAR": 4,
}


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _clean_heading_text(raw_doc: str) -> tuple[str, str]:
    """Separate clean heading text from any legacy chapter_notes_buffer table contamination."""
    if not raw_doc:
        return "", ""
    # Strip trailing ITC-HS table headers/rows accidentally appended in v0.1-v0.75
    parts = re.split(r"Tariff Item\s+Description of goods", raw_doc, maxsplit=1, flags=re.I)
    clean_part = parts[0].strip()
    if "Notes:" in clean_part:
        head_part, notes_part = clean_part.split("Notes:", 1)
        return head_part.strip().rstrip("."), notes_part.strip()
    return clean_part, ""


def _infer_ruling_authority(source_file: str, text: str) -> tuple[str, int]:
    override = _RULING_OVERRIDES.get(source_file, {})
    if override.get("authority"):
        auth = override["authority"]
        return auth, _AUTHORITY_RANK.get(auth, 5)
    upper = (text[:2000] + " " + source_file).upper()
    if "SUPREME COURT" in upper or "SLP" in upper or "CIVIL APPEAL" in upper:
        return "Supreme Court", 1
    if "HIGH COURT" in upper or re.search(r"\b(WP|WA|CWP|CMA)\b", upper):
        return "High Court", 2
    if "CESTAT" in upper or "APPELLATE TRIBUNAL" in upper:
        return "CESTAT", 3
    if "CAAR" in upper or "ADVANCE RULING" in upper:
        return "CAAR", 4
    return "CAAR", 4


class VectorDBClient:
    """Hybrid Vector + Hierarchical SQLite/BM25 Client for Indian ITC-HS ChromaDB."""

    def __init__(
        self,
        base_url: str = VECTOR_DB_BASE_URL,
        default_collection: str | None = DEFAULT_COLLECTION,
        persist_dir: str = CHROMA_PERSIST_DIR,
        embed_model_name: str = EMBED_MODEL_NAME,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.default_collection = default_collection or "itc_tariff_items"
        self.persist_dir = persist_dir
        self.embed_model_name = embed_model_name
        self.timeout = timeout

        self._chroma_http_client = self._create_http_client() if self.base_url else None
        self._chroma_persistent_client = None
        self._embedder = None

        self.collection_counts: dict[str, int] = {}
        self.pipeline_version: str = "v0_3"
        self.records_by_collection: dict[str, list[dict[str, Any]]] = {}
        self.bm25_by_collection: dict[str, Any] = {}
        self.headings_by_code: dict[str, dict[str, Any]] = {}
        self.subheadings_by_code: dict[str, dict[str, Any]] = {}
        self.items_by_code: dict[str, dict[str, Any]] = {}

        self._load_local_store()

    def _create_http_client(self) -> Any | None:
        if chromadb is None or not self.base_url:
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
        except Exception:
            return None

    def _get_embedder(self) -> Any | None:
        if not ENABLE_LOCAL_EMBEDDINGS:
            return None
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(self.embed_model_name)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to load SentenceTransformer %s: %s", self.embed_model_name, exc)
                return None
        return self._embedder

    def _get_persistent_collection(self, name: str) -> Any | None:
        if chromadb is None or not Path(self.persist_dir, "chroma.sqlite3").exists():
            return None
        try:
            if self._chroma_persistent_client is None:
                self._chroma_persistent_client = chromadb.PersistentClient(path=self.persist_dir)
            return self._chroma_persistent_client.get_collection(name)
        except Exception:
            return None

    def _load_local_store(self) -> None:
        sqlite_path = Path(self.persist_dir) / "chroma.sqlite3"
        if not sqlite_path.exists():
            logger.info("No local chroma.sqlite3 found at %s", sqlite_path)
            return

        try:
            conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        except Exception as exc:
            logger.warning("Failed to open %s read-only: %s", sqlite_path, exc)
            return

        try:
            colls = conn.execute("SELECT id, name FROM collections").fetchall()
            coll_id_to_name = {row["id"]: row["name"] for row in colls}

            # Detect pipeline_version & embed_model from collection_metadata if present (v0_3)
            meta_rows = conn.execute("SELECT key, str_value FROM collection_metadata").fetchall()
            meta_map = {r["key"]: r["str_value"] for r in meta_rows if r["str_value"]}
            if "pipeline_version" in meta_map:
                self.pipeline_version = meta_map["pipeline_version"]
            else:
                self.pipeline_version = "v0.7x-reconstructed"
            if "embed_model" in meta_map:
                self.embed_model_name = meta_map["embed_model"]

            seg_rows = conn.execute(
                "SELECT id, collection FROM segments WHERE scope = 'METADATA'"
            ).fetchall()
            seg_to_coll = {row["id"]: coll_id_to_name.get(row["collection"]) for row in seg_rows}

            raw_records: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in coll_id_to_name.values()}

            emb_rows = conn.execute("SELECT id, segment_id, embedding_id FROM embeddings").fetchall()
            emb_id_map: dict[int, tuple[str, str]] = {}
            for row in emb_rows:
                cname = seg_to_coll.get(row["segment_id"])
                if not cname:
                    continue
                emb_id_map[row["id"]] = (cname, row["embedding_id"])
                raw_records[cname][row["id"]] = {"id": row["embedding_id"], "content": "", "metadata": {}}

            em_rows = conn.execute(
                "SELECT id, key, string_value, int_value, float_value FROM embedding_metadata"
            ).fetchall()
            for row in em_rows:
                target = emb_id_map.get(row["id"])
                if not target:
                    continue
                cname, _ = target
                rec = raw_records[cname][row["id"]]
                val = (
                    row["string_value"]
                    if row["string_value"] is not None
                    else (row["int_value"] if row["int_value"] is not None else row["float_value"])
                )
                if row["key"] == "chroma:document":
                    rec["content"] = str(val or "")
                else:
                    rec["metadata"][row["key"]] = val
        finally:
            conn.close()

        # Index headings and subheadings first for lineage reconstruction
        for rec in raw_records.get("itc_headings", {}).values():
            meta = rec["metadata"]
            code = str(meta.get("code") or "")
            clean_head, notes_txt = _clean_heading_text(rec["content"])
            meta.setdefault("chapter", code[:2])
            if not meta.get("description") and clean_head:
                meta["description"] = clean_head
            rec["clean_content"] = clean_head
            rec["extracted_notes"] = notes_txt
            self.headings_by_code[code] = rec

        for rec in raw_records.get("itc_subheadings", {}).values():
            meta = rec["metadata"]
            code = str(meta.get("code") or "")
            self.subheadings_by_code[code] = rec

        # Reconstruct clean lineage for tariff items & subheadings
        for cname, recs_dict in raw_records.items():
            cleaned_list: list[dict[str, Any]] = []
            for rec in recs_dict.values():
                meta = rec["metadata"]
                if cname == "itc_tariff_items":
                    code = str(meta.get("code") or "")
                    h_code = str(meta.get("heading_code") or code[:4])
                    p_code = str(meta.get("parent_code") or code[:6])
                    h_rec = self.headings_by_code.get(h_code)
                    s_rec = self.subheadings_by_code.get(p_code) or self.subheadings_by_code.get(code[:6])
                    h_desc = (h_rec["metadata"].get("description") if h_rec else "") or ""
                    s_desc = (s_rec["metadata"].get("description") if s_rec else "") or ""
                    item_desc = str(meta.get("description") or "")
                    if " > " not in rec["content"]:
                        lineage_parts = [p for p in (h_desc, s_desc if s_desc and s_desc != item_desc else "", item_desc) if p]
                        lineage = " > ".join(lineage_parts)
                        meta["path_description"] = lineage
                        rec["content"] = (
                            f"Tariff item {code} | Chapter {code[:2]} | {lineage}"
                            + (f" | Unit: {meta.get('unit')}" if meta.get("unit") else "")
                            + (f" | Duty: {meta.get('duty_rate')}" if meta.get("duty_rate") else "")
                        )
                    self.items_by_code[code] = rec
                elif cname == "itc_headings":
                    rec["content"] = rec.get("clean_content") or rec["content"]
                elif cname == "itc_rulings":
                    src = str(meta.get("source_file") or "")
                    if src in _EXCLUDED_RULING_FILES:
                        continue
                    filtered_txt = filter_english_lines(rec["content"])
                    if len(filtered_txt) < 40:
                        continue
                    rec["content"] = filtered_txt
                    auth, rank = _infer_ruling_authority(src, filtered_txt)
                    meta["authority"] = meta.get("authority") or auth
                    meta["precedence_rank"] = int(meta.get("precedence_rank") or rank)
                    if src in _RULING_OVERRIDES:
                        ov = _RULING_OVERRIDES[src]
                        meta["hs_codes"] = ov.get("hs_codes", "")
                        meta["hs_chapter_primary"] = meta["hs_codes"][:2]
                        meta["hs_heading_primary"] = meta["hs_codes"][:4]

                cleaned_list.append(rec)

            self.records_by_collection[cname] = cleaned_list
            self.collection_counts[cname] = len(cleaned_list)
            if BM25Okapi is not None and cleaned_list:
                corpus_tokens = [_tokenize(r["content"]) for r in cleaned_list]
                self.bm25_by_collection[cname] = BM25Okapi(corpus_tokens)

        # Synthesize clean itc_notes collection from heading notes if v0_3 itc_notes is absent
        if "itc_notes" not in self.records_by_collection or not self.records_by_collection["itc_notes"]:
            synth_notes: list[dict[str, Any]] = []
            seen_note_keys: set[str] = set()
            for code, h_rec in self.headings_by_code.items():
                nt = (h_rec.get("extracted_notes") or "").strip()
                if nt and len(nt) > 20:
                    ch = code[:2]
                    sig = f"{ch}:{nt[:120]}"
                    if sig not in seen_note_keys:
                        seen_note_keys.add(sig)
                        synth_notes.append(
                            {
                                "id": f"synth_note_{ch}_{len(synth_notes)}",
                                "content": f"Chapter {ch} Legal Notes: {nt}",
                                "metadata": {"scope": "chapter", "chapter": ch, "kind": "chapter notes"},
                            }
                        )
            if synth_notes:
                self.records_by_collection["itc_notes"] = synth_notes
                self.collection_counts["itc_notes"] = len(synth_notes)
                if BM25Okapi is not None:
                    self.bm25_by_collection["itc_notes"] = BM25Okapi(
                        [_tokenize(r["content"]) for r in synth_notes]
                    )

    def lookup_code(self, code: str) -> dict[str, Any] | None:
        digits = re.sub(r"\D", "", code or "")
        if len(digits) == 8 and digits in self.items_by_code:
            return self.items_by_code[digits]
        if len(digits) == 6 and digits in self.subheadings_by_code:
            return self.subheadings_by_code[digits]
        if len(digits) == 4 and digits in self.headings_by_code:
            return self.headings_by_code[digits]
        return None

    def _matches_filters(self, metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        for key, cond in filters.items():
            val = str(metadata.get(key, ""))
            if isinstance(cond, dict) and "$in" in cond:
                if val not in [str(x) for x in cond["$in"]]:
                    return False
            elif val != str(cond):
                return False
        return True

    def _local_hybrid_search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        records = self.records_by_collection.get(collection_name) or []
        if not records:
            return []

        q_tokens = _tokenize(query)
        bm25 = self.bm25_by_collection.get(collection_name)
        bm25_scores: list[float]
        if bm25 is not None and q_tokens:
            bm25_scores = list(bm25.get_scores(q_tokens))
        else:
            q_set = set(q_tokens)
            bm25_scores = [
                float(len(q_set & set(_tokenize(r["content"])))) for r in records
            ]

        # Optional dense Chroma embedding scores fused via RRF when local embeddings enabled
        dense_ranks: dict[str, int] = {}
        embedder = self._get_embedder()
        coll_handle = self._get_persistent_collection(collection_name) if embedder is not None else None
        if embedder is not None and coll_handle is not None:
            try:
                q_emb = embedder.encode([query], normalize_embeddings=True).tolist()
                dense_res = coll_handle.query(
                    query_embeddings=q_emb,
                    n_results=min(top_k * 3, 25),
                    where=filters,
                    include=["distances"],
                )
                for rank_idx, doc_id in enumerate((dense_res.get("ids") or [[]])[0]):
                    dense_ranks[doc_id] = rank_idx + 1
            except Exception as exc:  # pragma: no cover
                logger.warning("Dense query failed on %s: %s", collection_name, exc)

        # Rank filtered candidates
        scored_indices = [
            (idx, bm25_scores[idx])
            for idx, rec in enumerate(records)
            if self._matches_filters(rec["metadata"], filters)
        ]
        scored_indices.sort(key=lambda pair: pair[1], reverse=True)
        max_bm25 = max((s for _, s in scored_indices), default=0.0) or 1.0

        results: list[dict[str, Any]] = []
        top_pool = scored_indices[: max(top_k * 4, 25)]
        for bm25_rank, (idx, raw_bm25) in enumerate(top_pool, start=1):
            rec = records[idx]
            norm_bm25 = min(1.0, raw_bm25 / max(max_bm25, 1e-6)) if raw_bm25 > 0 else 0.0
            d_rank = dense_ranks.get(rec["id"])
            if d_rank is not None:
                rrf = (1.0 / (50 + bm25_rank)) + (1.0 / (50 + d_rank))
                score = min(0.99, round(rrf * 25.0, 4))
            else:
                score = round(norm_bm25, 4)

            if score <= 0.0 and not dense_ranks and not filters:
                continue

            results.append(
                {
                    "id": rec["id"],
                    "content": rec["content"],
                    "score": score,
                    "metadata": dict(rec["metadata"]),
                }
            )

        results.sort(key=lambda r: r.get("score") or 0.0, reverse=True)
        # Fallback if filters were given and all BM25 scores were 0
        if not results and filters:
            for idx, _ in scored_indices[:top_k]:
                rec = records[idx]
                results.append(
                    {
                        "id": rec["id"],
                        "content": rec["content"],
                        "score": 0.1,
                        "metadata": dict(rec["metadata"]),
                    }
                )
        return results[:top_k]

    def retrieve_hierarchical_bundle(
        self,
        query: str,
        top_headings: int = 5,
        top_items: int = 8,
        top_notes: int = 4,
        top_rulings: int = 3,
    ) -> dict[str, Any]:
        """Stage-1 Heading/Subheading -> Stage-2 Notes & Tariff Items -> Stage-3 GRIs & Precedents."""
        # Exact HS code mention shortcut
        code_hits: list[dict[str, Any]] = []
        for tok in re.findall(r"\b\d{4}(?:\s?\d{2}){0,2}\b", query):
            rec = self.lookup_code(tok)
            if rec:
                code_hits.append({"id": rec["id"], "content": rec["content"], "score": 1.0, "metadata": dict(rec["metadata"])})

        headings = self._local_hybrid_search(query, "itc_headings", top_k=top_headings)
        subheadings = self._local_hybrid_search(query, "itc_subheadings", top_k=4)

        candidate_headings = list(
            dict.fromkeys(
                [str(h["metadata"].get("code", "")) for h in headings if h["metadata"].get("code")]
                + [str(s["metadata"].get("heading_code", "")) for s in subheadings if s["metadata"].get("heading_code")]
            )
        )
        candidate_chapters = list(dict.fromkeys([h[:2] for h in candidate_headings if len(h) >= 2]))

        filtered_items = (
            self._local_hybrid_search(
                query,
                "itc_tariff_items",
                top_k=top_items,
                filters={"heading_code": {"$in": candidate_headings}},
            )
            if candidate_headings
            else []
        )
        global_items = self._local_hybrid_search(query, "itc_tariff_items", top_k=top_items)

        merged_items: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in code_hits + filtered_items + global_items:
            if item["id"] not in seen_ids and item["metadata"].get("level", "tariff_item") == "tariff_item":
                seen_ids.add(item["id"])
                merged_items.append(item)
        merged_items.sort(key=lambda r: r.get("score") or 0.0, reverse=True)
        merged_items = merged_items[:top_items]

        # Statutory notes
        chapter_notes = (
            self._local_hybrid_search(
                query,
                "itc_notes",
                top_k=top_notes,
                filters={"chapter": {"$in": candidate_chapters}},
            )
            if candidate_chapters
            else []
        )
        semantic_notes = self._local_hybrid_search(query, "itc_notes", top_k=top_notes)
        notes = []
        seen_note_ids: set[str] = set()
        for n in chapter_notes + semantic_notes:
            if n["id"] not in seen_note_ids:
                seen_note_ids.add(n["id"])
                notes.append(n)
        notes = notes[:top_notes]

        gris = self._local_hybrid_search(query or "classification heading notes", "itc_gris", top_k=3)
        if not gris and "itc_gris" in self.records_by_collection:
            gris = [
                {"id": r["id"], "content": r["content"], "score": 0.5, "metadata": dict(r["metadata"])}
                for r in self.records_by_collection["itc_gris"][:2]
            ]

        rulings = self._local_hybrid_search(query, "itc_rulings", top_k=top_rulings)
        # Deterministic conflict resolution ordering: Supreme Court (1) > High Court (2) > CESTAT (3) > CAAR (4)
        rulings.sort(
            key=lambda r: (
                int(r["metadata"].get("precedence_rank") or 5),
                -(r.get("score") or 0.0),
            )
        )

        max_item_score = max((float(x.get("score") or 0.0) for x in merged_items), default=0.0)
        return {
            "headings": headings,
            "subheadings": subheadings,
            "items": merged_items,
            "notes": notes,
            "gris": gris,
            "rulings": rulings,
            "max_similarity": max_item_score,
        }

    def search(
        self,
        query: str,
        collection: str | None = None,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        target_collection = collection or self.default_collection

        # 1. Local persisted ChromaDB SQLite/hybrid search
        if target_collection in self.records_by_collection:
            local_hits = self._local_hybrid_search(
                query=query,
                collection_name=target_collection,
                top_k=top_k,
                filters=filters,
            )
            if local_hits:
                return local_hits

        # 2. Optional remote Chroma HTTP server fallback
        if self._chroma_http_client is not None and target_collection:
            try:
                collection_handle = self._chroma_http_client.get_collection(target_collection)
                result = collection_handle.query(
                    query_texts=[query],
                    n_results=min(top_k, 10),
                    include=["documents", "metadatas", "distances"],
                    where=filters,
                )
                return self._normalize_chroma_response(result)
            except Exception as exc:  # pragma: no cover
                last_error: Exception | None = exc
        else:
            last_error = None

        # 3. Generic HTTP fallback endpoints when VECTOR_DB_BASE_URL is configured
        if self.base_url:
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
                except requests.RequestException as exc:  # pragma: no cover
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
                return normalized_documents

        if last_error is not None:
            raise VectorDBConnectionError(
                "Unable to reach the vector database. "
                f"Checked local store ({self.persist_dir}) and remote endpoints. Last error: {last_error}"
            )
        return []

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
