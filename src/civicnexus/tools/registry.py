from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Callable
from pydantic import BaseModel, Field
from civicnexus.domain.results import ToolRequest


class ToolError(RuntimeError):
    pass


class ToolContext(BaseModel):
    task_id: str
    user_id: str
    session_id: str


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any]
    result: Any = None
    success: bool
    duration_ms: float
    error: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    args_model: type[BaseModel]
    handler: Callable[[BaseModel, ToolContext], Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        args_model: type[BaseModel],
        handler: Callable[[BaseModel, ToolContext], Any],
    ) -> None:
        self._tools[name] = ToolSpec(args_model=args_model, handler=handler)

    def execute(
        self,
        request: ToolRequest,
        context: ToolContext,
        *,
        allowed_tools: set[str] | None = None,
    ) -> ToolCallRecord:
        started = time.perf_counter()
        try:
            if allowed_tools is not None and request.name not in allowed_tools:
                raise ToolError(f"Skill does not allow tool '{request.name}'.")
            spec = self._tools.get(request.name)
            if not spec:
                raise ToolError(f"Unknown tool '{request.name}'.")
            arguments = dict(request.arguments)
            if request.idempotency_key:
                arguments.setdefault("idempotency_key", request.idempotency_key)
            parsed = spec.args_model.model_validate(arguments)
            result = spec.handler(parsed, context)
            return ToolCallRecord(
                name=request.name,
                arguments=arguments,
                result=result,
                success=True,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except (ValueError, ToolError) as exc:
            return ToolCallRecord(
                name=request.name,
                arguments=request.arguments,
                success=False,
                error=str(exc),
                duration_ms=(time.perf_counter() - started) * 1000,
            )


class CaseSearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)


class AddressArgs(BaseModel):
    address: str = Field(min_length=2, max_length=120)


class MemoryLookupArgs(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    top_k: int = Field(default=3, ge=1, le=10)


class CreateCaseArgs(BaseModel):
    category: str
    location: str
    department: str
    priority: str = "普通"
    summary: str = ""
    idempotency_key: str
