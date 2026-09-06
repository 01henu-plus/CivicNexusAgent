from __future__ import annotations
from hashlib import sha256
from typing import Any, Callable
from civicnexus.retrieval import (
    EmbeddingAPIError,
    EmbeddingConfigurationError,
    RetrievalHit,
    load_hybrid_retriever_from_settings,
)
from civicnexus.tools.registry import (
    AddressArgs,
    CaseSearchArgs,
    CreateCaseArgs,
    MemoryLookupArgs,
    ToolContext,
    ToolRegistry,
)


def build_tool_registry(
    *,
    search: Callable[[str, int], list[Any]] | None = None,
    memory_lookup: Callable[[ToolContext, str, int], list[dict[str, Any]]] | None = None,
    create_case: Callable[[ToolContext, CreateCaseArgs], dict[str, Any]] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()

    def case_search(args: CaseSearchArgs, _: ToolContext) -> list[dict[str, Any]]:
        if search:
            hits = search(args.query, args.top_k)
        else:
            try:
                hits = _online_search(args.query, args.top_k)
            except (
                EmbeddingAPIError,
                EmbeddingConfigurationError,
                FileNotFoundError,
                RuntimeError,
                ValueError,
            ):
                hits = _demo_search(args.query, args.top_k)
        return [_hit_dict(hit) for hit in hits]

    def validate_address(args: AddressArgs, _: ToolContext) -> dict[str, Any]:
        normalized = " ".join(args.address.strip().split())
        return {"valid": len(normalized) >= 3, "normalized": normalized}

    def lookup(args: MemoryLookupArgs, context: ToolContext) -> list[dict[str, Any]]:
        return memory_lookup(context, args.query, args.top_k) if memory_lookup else []

    def create(args: CreateCaseArgs, context: ToolContext) -> dict[str, Any]:
        if create_case:
            return create_case(context, args)
        stable = sha256(f"{context.task_id}:{args.idempotency_key}".encode()).hexdigest()[:12]
        return {"case_id": f"LOCAL-{stable.upper()}", "dry_run": True}

    registry.register("case_search", CaseSearchArgs, case_search)
    registry.register("validate_address", AddressArgs, validate_address)
    registry.register("memory_lookup", MemoryLookupArgs, lookup)
    registry.register("create_local_case", CreateCaseArgs, create)
    return registry


def _online_search(query: str, top_k: int) -> list[RetrievalHit]:
    return load_hybrid_retriever_from_settings().search(query, final_top_k=top_k)


_DEMO_CASES = [
    {
        "case_id": "demo-sewer-01",
        "category_zh": "污水与下水道",
        "status_zh": "已处理",
        "keywords_zh": "污水、下水道、堵塞",
        "source": "demo",
    },
    {
        "case_id": "demo-drain-01",
        "category_zh": "排水、积水与洪涝",
        "status_zh": "已处理",
        "keywords_zh": "排水、积水、内涝",
        "source": "demo",
    },
    {
        "case_id": "demo-garbage-01",
        "category_zh": "垃圾处理",
        "status_zh": "已处理",
        "keywords_zh": "垃圾、清运、收集",
        "source": "demo",
    },
    {
        "case_id": "demo-road-01",
        "category_zh": "道路维护",
        "status_zh": "已处理",
        "keywords_zh": "道路、坑洞、路面",
        "source": "demo",
    },
]


def _demo_search(query: str, top_k: int) -> list[dict[str, Any]]:
    terms = set(query)
    ranked = sorted(
        _DEMO_CASES,
        key=lambda item: (
            -len(terms.intersection(set(item["keywords_zh"] + item["category_zh"]))),
            item["case_id"],
        ),
    )
    return ranked[:top_k]


def _hit_dict(hit: Any) -> dict[str, Any]:
    if isinstance(hit, RetrievalHit):
        return {
            "case_id": hit.case_id,
            "score": hit.score,
            "source": hit.source,
            **hit.metadata,
        }
    return dict(hit)
