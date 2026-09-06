"""Unified BGE-M3, BM25, and hybrid retrieval implementation."""

from __future__ import annotations
import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import httpx
import jieba
from rank_bm25 import BM25Okapi


class EmbeddingConfigurationError(ValueError):
    """Raised when the external embedding service is not configured."""


class EmbeddingAPIError(RuntimeError):
    """Raised when an embedding service returns an unusable response."""


@dataclass(frozen=True)
class RetrievalDocument:
    """One historical case represented for vector and keyword retrieval."""

    case_id: str
    vector_text: str
    keyword_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    case_id: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    document: str = ""


class BGEM3Encoder:
    """OpenAI-compatible external BGE-M3 embedding adapter."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        if not base_url.strip():
            raise EmbeddingConfigurationError("EMBEDDING_BASE_URL must be configured.")
        if not api_key.strip():
            raise EmbeddingConfigurationError("EMBEDDING_API_KEY must be configured.")
        self.endpoint = f"{base_url.rstrip('/')}/embeddings"
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed one batch while preserving the supplied text order."""
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not one string.")
        if not texts:
            return []
        if any((not isinstance(text, str) or not text.strip() for text in texts)):
            raise ValueError("Every text passed to the embedding service must be non-empty.")
        payload = {"model": self.model_name, "input": list(texts), "encoding_format": "float"}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response: httpx.Response | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(self.endpoint, json=payload, headers=headers)
                    response.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                if attempt == self.max_retries:
                    raise EmbeddingAPIError(
                        f"External embedding service returned HTTP {exc.response.status_code}."
                    ) from exc
            except httpx.HTTPError as exc:
                if attempt == self.max_retries:
                    raise EmbeddingAPIError("External embedding request failed.") from exc
            time.sleep(2**attempt)
        if response is None:
            raise EmbeddingAPIError("External embedding request failed.")
        try:
            response_payload = response.json()
        except ValueError as exc:
            raise EmbeddingAPIError("Embedding service did not return JSON.") from exc
        return self._parse_embeddings(response_payload, expected_count=len(texts))

    @staticmethod
    def _parse_embeddings(payload: Any, *, expected_count: int) -> list[list[float]]:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise EmbeddingAPIError("Embedding response does not contain a data list.")
        items = payload["data"]
        if len(items) != expected_count:
            raise EmbeddingAPIError("Embedding response count does not match request count.")
        try:
            ordered_items = sorted(items, key=lambda item: item["index"])
            indexes = [item["index"] for item in ordered_items]
            embeddings = [item["embedding"] for item in ordered_items]
        except (KeyError, TypeError):
            raise EmbeddingAPIError("Embedding response has an invalid item format.") from None
        if indexes != list(range(expected_count)):
            raise EmbeddingAPIError("Embedding response indexes do not match request order.")
        if any(
            (
                not isinstance(vector, list)
                or not vector
                or any(
                    (
                        isinstance(value, bool) or not isinstance(value, int | float)
                        for value in vector
                    )
                )
                for vector in embeddings
            )
        ):
            raise EmbeddingAPIError("Embedding response contains an invalid vector.")
        return [[float(value) for value in vector] for vector in embeddings]


class BGEM3Retriever:
    """Persistent ChromaDB retriever backed by externally generated BGE-M3 vectors."""

    def __init__(
        self,
        encoder: BGEM3Encoder,
        *,
        persist_directory: str | Path | None = None,
        collection_name: str = "civic_cases",
        chroma_host: str | None = None,
        chroma_port: int = 8000,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb must be installed before vector retrieval can be used."
            ) from exc
        self.encoder = encoder
        self.collection_name = collection_name
        if chroma_host:
            self._client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        else:
            if persist_directory is None:
                raise ValueError("persist_directory is required for embedded Chroma mode.")
            self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._create_collection()

    @property
    def document_count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Remove and recreate this project's vector collection."""
        try:
            self._client.delete_collection(name=self.collection_name)
        except ValueError:
            pass
        self._collection = self._create_collection()

    def upsert(self, documents: Sequence[RetrievalDocument], *, batch_size: int = 64) -> int:
        """Generate vectors in batches and persist only case IDs not already indexed."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if len({document.case_id for document in documents}) != len(documents):
            raise ValueError("Vector documents must have unique case_id values.")
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            existing = self._collection.get(
                ids=[document.case_id for document in batch], include=[]
            )
            existing_ids = set(existing["ids"])
            pending_documents = [
                document for document in batch if document.case_id not in existing_ids
            ]
            if not pending_documents:
                continue
            embeddings = self.encoder.embed(
                [document.vector_text for document in pending_documents]
            )
            self._collection.upsert(
                ids=[document.case_id for document in pending_documents],
                embeddings=embeddings,
                documents=[document.vector_text for document in pending_documents],
                metadatas=[_chroma_metadata(document) for document in pending_documents],
            )
        return len(documents)

    def search(self, query: str, *, top_k: int = 8) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not query.strip() or self.document_count == 0:
            return []
        result = self._collection.query(
            query_embeddings=self.encoder.embed([query]),
            n_results=min(top_k, self.document_count),
            include=["metadatas", "documents", "distances"],
        )
        case_ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]
        hits: list[RetrievalHit] = []
        for case_id, distance, metadata, document in zip(
            case_ids, distances, metadatas, documents, strict=True
        ):
            if not isinstance(case_id, str) or not isinstance(distance, int | float):
                continue
            hits.append(
                RetrievalHit(
                    case_id=case_id,
                    score=max(0.0, 1.0 - float(distance)),
                    source="bge_m3",
                    metadata=dict(metadata or {}),
                    document=document or "",
                )
            )
        return hits

    def _create_collection(self) -> Any:
        return self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )


class BM25Retriever:
    """Persistent local BM25 index over the Chinese standardized case text."""

    def __init__(self, documents: Sequence[RetrievalDocument]) -> None:
        if len({document.case_id for document in documents}) != len(documents):
            raise ValueError("BM25 documents must have unique case_id values.")
        self.documents = list(documents)
        self._tokenized_corpus = [tokenize_zh(document.keyword_text) for document in self.documents]
        self._index = BM25Okapi(self._tokenized_corpus) if self.documents else None

    @property
    def document_count(self) -> int:
        return len(self.documents)

    def search(self, query: str, *, top_k: int = 8) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not self._index or not query.strip():
            return []
        query_tokens = tokenize_zh(query)
        if not query_tokens:
            return []
        scores = self._index.get_scores(query_tokens)
        ranked = sorted(
            ((index, float(score)) for index, score in enumerate(scores) if score != 0),
            key=lambda item: (-item[1], self.documents[item[0]].case_id),
        )
        if not ranked:
            query_terms = set(query_tokens)
            ranked = sorted(
                (
                    (index, float(len(query_terms.intersection(tokens))))
                    for index, tokens in enumerate(self._tokenized_corpus)
                    if query_terms.intersection(tokens)
                ),
                key=lambda item: (-item[1], self.documents[item[0]].case_id),
            )
        return [
            RetrievalHit(
                case_id=self.documents[index].case_id,
                score=score,
                source="bm25",
                metadata=dict(self.documents[index].metadata),
                document=self.documents[index].keyword_text,
            )
            for index, score in ranked[:top_k]
        ]

    def save(self, path: str | Path) -> None:
        """Persist source documents; the deterministic BM25 index is rebuilt when loaded."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "documents": [asdict(document) for document in self.documents],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BM25Retriever:
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            raw_documents = payload["documents"]
        except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to load BM25 index from {source}.") from exc
        if not isinstance(raw_documents, list):
            raise ValueError(f"BM25 index at {source} has an invalid document list.")
        try:
            documents = [
                RetrievalDocument(
                    case_id=item["case_id"],
                    vector_text=item["vector_text"],
                    keyword_text=item["keyword_text"],
                    metadata=item.get("metadata", {}),
                )
                for item in raw_documents
            ]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"BM25 index at {source} contains an invalid document.") from exc
        return cls(documents)


class HybridRetriever:
    """Fuse BGE-M3 and BM25 candidates with weighted reciprocal-rank fusion."""

    def __init__(
        self,
        vector_retriever: BGEM3Retriever,
        bm25_retriever: BM25Retriever,
        *,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> None:
        if vector_weight <= 0 or bm25_weight <= 0:
            raise ValueError("Hybrid retrieval weights must be positive.")
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1.")
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    def search(
        self, query: str, *, vector_top_k: int = 8, bm25_top_k: int = 8, final_top_k: int = 5
    ) -> list[RetrievalHit]:
        vector_hits = self.vector_retriever.search(query, top_k=vector_top_k)
        bm25_hits = self.bm25_retriever.search(query, top_k=bm25_top_k)
        return self.merge(vector_hits, bm25_hits, final_top_k=final_top_k)

    def merge(
        self,
        vector_hits: Sequence[RetrievalHit],
        bm25_hits: Sequence[RetrievalHit],
        *,
        final_top_k: int = 5,
    ) -> list[RetrievalHit]:
        if final_top_k < 1:
            raise ValueError("final_top_k must be at least 1.")
        merged: dict[str, dict[str, Any]] = {}
        self._add_ranked_hits(merged, vector_hits, self.vector_weight)
        self._add_ranked_hits(merged, bm25_hits, self.bm25_weight)
        hits = [
            RetrievalHit(
                case_id=case_id,
                score=entry["score"],
                source="+".join(sorted(entry["sources"])),
                metadata=entry["metadata"],
                document=entry["document"],
            )
            for case_id, entry in merged.items()
        ]
        return sorted(hits, key=lambda hit: (-hit.score, hit.case_id))[:final_top_k]

    def _add_ranked_hits(
        self, merged: dict[str, dict[str, Any]], hits: Sequence[RetrievalHit], weight: float
    ) -> None:
        seen_case_ids: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            if hit.case_id in seen_case_ids:
                continue
            seen_case_ids.add(hit.case_id)
            entry = merged.setdefault(
                hit.case_id,
                {
                    "score": 0.0,
                    "sources": set(),
                    "metadata": dict(hit.metadata),
                    "document": hit.document,
                },
            )
            entry["score"] += weight / (self.rrf_k + rank)
            entry["sources"].add(hit.source)
            if not entry["metadata"] and hit.metadata:
                entry["metadata"] = dict(hit.metadata)
            if not entry["document"] and hit.document:
                entry["document"] = hit.document


def load_hybrid_retriever_from_settings() -> HybridRetriever:
    """Create the online retriever from persisted indexes and environment settings."""
    from civicnexus.core.config import get_settings

    settings = get_settings()
    encoder = BGEM3Encoder(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key.get_secret_value(),
        model_name=settings.embedding_model,
        timeout_seconds=settings.embedding_timeout_seconds,
        max_retries=settings.embedding_max_retries,
    )
    vector_retriever = BGEM3Retriever(
        encoder,
        persist_directory=settings.chroma_persist_directory,
        collection_name=settings.retrieval_collection_name,
        chroma_host=settings.chroma_host or None,
        chroma_port=settings.chroma_port,
    )
    bm25_retriever = BM25Retriever.load(settings.bm25_index_path)
    return HybridRetriever(vector_retriever, bm25_retriever)


_STOP_TOKENS = {"的", "了", "和", "在", "是", "有", "这", "这里", "一个", "一下", "需要"}
_MEANINGFUL_TOKEN = re.compile("[a-z0-9\\u4e00-\\u9fff]", re.IGNORECASE)


def tokenize_zh(text: str) -> list[str]:
    """Segment Chinese case text while retaining English terms and model identifiers."""
    if not isinstance(text, str) or not text.strip():
        return []
    tokens: list[str] = []
    for token in jieba.lcut(text.lower(), HMM=False):
        normalized = token.strip()
        if normalized and normalized not in _STOP_TOKENS and _MEANINGFUL_TOKEN.search(normalized):
            tokens.append(normalized)
    return tokens


def _chroma_metadata(document: RetrievalDocument) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {"case_id": document.case_id}
    for key, value in document.metadata.items():
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            metadata[str(key)] = value
        else:
            metadata[str(key)] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return metadata
