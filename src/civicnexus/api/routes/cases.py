from __future__ import annotations
from functools import lru_cache
from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from civicnexus.retrieval import (
    EmbeddingAPIError,
    EmbeddingConfigurationError,
    RetrievalHit,
    load_hybrid_retriever_from_settings,
)

router = APIRouter()


class CaseSearchRequest(BaseModel):
    message: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class CaseEvidence(BaseModel):
    case_id: str
    score: float
    evidence_source: str
    category_zh: str | None = None
    status_zh: str | None = None
    keywords_zh: str | None = None


class CaseSearchResponse(BaseModel):
    message_zh: str
    evidence: list[CaseEvidence]


@router.post("/cases/search", response_model=CaseSearchResponse)
def search_historical_cases(request: CaseSearchRequest) -> CaseSearchResponse:
    try:
        hits = _get_retriever().search(request.message, final_top_k=request.top_k)
    except EmbeddingConfigurationError as exc:
        raise HTTPException(status_code=503, detail="尚未配置 BGE-M3 Embedding 服务。") from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="历史案例索引尚未构建。") from exc
    except EmbeddingAPIError as exc:
        raise HTTPException(status_code=502, detail="BGE-M3 Embedding 服务调用失败。") from exc
    return CaseSearchResponse(
        message_zh="已找到相关历史案例。" if hits else "未找到足够相关的历史案例。",
        evidence=[_to_evidence(hit) for hit in hits],
    )


@lru_cache
def _get_retriever() -> Any:
    return load_hybrid_retriever_from_settings()


def _to_evidence(hit: RetrievalHit) -> CaseEvidence:
    metadata = hit.metadata
    return CaseEvidence(
        case_id=hit.case_id,
        score=round(hit.score, 6),
        evidence_source=_source_name_zh(hit.source),
        category_zh=_metadata_text(metadata, "category_zh"),
        status_zh=_metadata_text(metadata, "status_zh"),
        keywords_zh=_metadata_text(metadata, "keywords_zh"),
    )


def _metadata_text(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) and value else None


def _source_name_zh(source: str) -> str:
    names = {
        "bge_m3": "语义检索",
        "bm25": "关键词检索",
    }
    return "、".join(names.get(item, item) for item in source.split("+"))
