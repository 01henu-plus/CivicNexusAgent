"""Build persistent retrieval indexes from CivicNexus historical cases."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any
from civicnexus.core.config import get_settings
from civicnexus.retrieval.retriever import (
    BGEM3Encoder,
    BGEM3Retriever,
    BM25Retriever,
    RetrievalDocument,
)

DEFAULT_DATA_PATH = Path(os.getenv("CITY_DATA_PATH", "city_data/cases_normalized.jsonl"))


def load_retrieval_documents(data_path: Path) -> list[RetrievalDocument]:
    """Read the retrieval split and retain only query-time case fields."""
    documents: list[RetrievalDocument] = []
    case_ids: set[str] = set()
    with data_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {data_path}:{line_number}.") from exc
            if record.get("dataset_split") != "retrieval":
                continue
            case_id = _required_text(record, "case_id", data_path, line_number)
            if case_id in case_ids:
                raise ValueError(f"Duplicate case_id '{case_id}' at {data_path}:{line_number}.")
            case_ids.add(case_id)
            documents.append(
                RetrievalDocument(
                    case_id=case_id,
                    vector_text=_required_text(record, "raw_text_en", data_path, line_number),
                    keyword_text=_required_text(
                        record, "retrieval_text_zh", data_path, line_number
                    ),
                    metadata=_build_metadata(record),
                )
            )
    if not documents:
        raise ValueError(f"No retrieval documents found in {data_path}.")
    return documents


def build_indexes(*, data_path: Path, reset_vector_index: bool = False) -> tuple[int, int]:
    """Build Chinese BM25 and BGE-M3 vector indexes, returning their document counts."""
    documents = load_retrieval_documents(data_path)
    settings = get_settings()
    bm25_retriever = BM25Retriever(documents)
    bm25_retriever.save(settings.bm25_index_path)
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
    if reset_vector_index:
        vector_retriever.reset()
    vector_retriever.upsert(documents, batch_size=settings.embedding_batch_size)
    return bm25_retriever.document_count, vector_retriever.document_count


def _required_text(
    record: dict[str, Any], field_name: str, data_path: Path, line_number: int
) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {field_name} at {data_path}:{line_number}.")
    return value.strip()


def _build_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "category_zh": record.get("category_zh"),
        "status_zh": record.get("status_zh"),
        "department": record.get("department"),
        "division": record.get("division"),
        "dept_div": record.get("dept_div"),
        "keywords_zh": "、".join(record.get("keywords_zh", [])),
        "source_layer": record.get("source_layer"),
        "source_id": record.get("source_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CivicNexus historical-case retrieval indexes."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--reset-vector-index",
        action="store_true",
        help="Delete the current ChromaDB collection before vector precomputation.",
    )
    arguments = parser.parse_args()
    bm25_count, vector_count = build_indexes(
        data_path=arguments.input,
        reset_vector_index=arguments.reset_vector_index,
    )
    print(f"BM25 索引已构建：{bm25_count} 条；BGE-M3 向量已写入：{vector_count} 条。")


if __name__ == "__main__":
    main()
