import pytest

from civicnexus.retrieval.retriever import (
    BGEM3Encoder,
    BM25Retriever,
    EmbeddingConfigurationError,
    HybridRetriever,
    RetrievalDocument,
    RetrievalHit,
)


def test_encoder_requires_external_api_configuration() -> None:
    with pytest.raises(EmbeddingConfigurationError, match="EMBEDDING_BASE_URL"):
        BGEM3Encoder(
            base_url="",
            api_key="test-key",
            model_name="BAAI/bge-m3",
        )


def test_parse_embeddings_restores_response_order() -> None:
    payload = {
        "data": [
            {"index": 1, "embedding": [0.3, 0.4]},
            {"index": 0, "embedding": [0.1, 0.2]},
        ]
    }

    embeddings = BGEM3Encoder._parse_embeddings(payload, expected_count=2)

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]


def test_bm25_returns_chinese_keyword_match() -> None:
    retriever = BM25Retriever(
        [
            RetrievalDocument(
                case_id="sewer-case",
                vector_text="SEWER BACKUP BLOCKED DRAIN",
                keyword_text="事项类别：污水与下水道；关键词：污水、下水道、堵塞",
            ),
            RetrievalDocument(
                case_id="garbage-case",
                vector_text="MISSED GARBAGE COLLECTION",
                keyword_text="事项类别：垃圾处理；关键词：垃圾、清运、收集",
            ),
        ]
    )

    hits = retriever.search("下水道污水堵塞", top_k=1)

    assert [hit.case_id for hit in hits] == ["sewer-case"]


def test_hybrid_merges_duplicate_cases_by_rank() -> None:
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.vector_weight = 1.0
    retriever.bm25_weight = 1.0
    retriever.rrf_k = 60

    hits = retriever.merge(
        [
            RetrievalHit(case_id="case-a", score=0.9, source="bge_m3"),
            RetrievalHit(case_id="case-b", score=0.8, source="bge_m3"),
        ],
        [
            RetrievalHit(case_id="case-b", score=3.0, source="bm25"),
            RetrievalHit(case_id="case-a", score=2.0, source="bm25"),
        ],
        final_top_k=2,
    )

    assert [hit.case_id for hit in hits] == ["case-a", "case-b"]
    assert hits[0].source == "bge_m3+bm25"
