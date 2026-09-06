from __future__ import annotations
import asyncio
import json
import time
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Callable
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from civicnexus.api.auth import get_principal, resolve_user_id
from civicnexus.api.services import get_service

router = APIRouter()


def _sse(event: str, payload: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


async def _stream(operation: Callable[[Callable[[Any], None]], Any], service: Any):
    queue: Queue[Any] = Queue()
    finished = Event()
    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            result["context"] = operation(queue.put)
        except Exception as exc:
            result["error"] = exc
        finally:
            finished.set()

    Thread(target=worker, daemon=True).start()
    yield _sse("status", {"state": "started"})
    heartbeat_at = time.monotonic()
    while not finished.is_set() or not queue.empty():
        try:
            trace = queue.get_nowait()
        except Empty:
            if time.monotonic() - heartbeat_at >= 10:
                yield _sse("status", {"state": "running"})
                heartbeat_at = time.monotonic()
            await asyncio.sleep(0.1)
            continue
        yield _sse("status", {"state": "running"})
    if "error" in result:
        yield _sse("error", {"message": "暂时无法回复，请稍后再试。"})
        return
    payload = service.user_view(result["context"])
    reply = payload.get("reply") or ""
    for index in range(0, len(reply), 24):
        yield _sse("delta", {"content": reply[index : index + 24]})
        await asyncio.sleep(0.01)
    yield _sse("complete", payload)


class TaskRequest(BaseModel):
    user_id: str | None = Field(default=None, min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=2, max_length=500)


@router.post("/tasks")
def create_task(
    request: TaskRequest, principal: dict[str, str] | None = Depends(get_principal)
) -> dict:
    user_id = resolve_user_id(request.user_id, principal)
    return get_service().user_view(
        get_service().create_task(user_id, request.session_id, request.message)
    )


@router.post("/tasks/stream")
async def create_task_stream(
    request: TaskRequest, principal: dict[str, str] | None = Depends(get_principal)
):
    user_id = resolve_user_id(request.user_id, principal)
    service = get_service()
    return StreamingResponse(
        _stream(
            lambda callback: service.create_task(
                user_id, request.session_id, request.message, on_trace=callback
            ),
            service,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/tasks/{task_id}/messages")
def add_message(
    task_id: str, request: TaskRequest, principal: dict[str, str] | None = Depends(get_principal)
) -> dict:
    try:
        context = get_service().add_message(
            task_id,
            resolve_user_id(request.user_id, principal),
            request.session_id,
            request.message,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="事项不存在。") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="无权访问该事项。") from exc
    return get_service().user_view(context)


@router.post("/tasks/{task_id}/messages/stream")
async def add_message_stream(
    task_id: str, request: TaskRequest, principal: dict[str, str] | None = Depends(get_principal)
):
    service = get_service()
    user_id = resolve_user_id(request.user_id, principal)
    try:
        context = service.load(task_id)
        if context.user_id != user_id or context.session_id != request.session_id:
            raise PermissionError("task does not belong to this session")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="事项不存在。") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="无权访问该事项。") from exc
    return StreamingResponse(
        _stream(
            lambda callback: service.add_message(
                task_id, user_id, request.session_id, request.message, on_trace=callback
            ),
            service,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/tasks/{task_id}/messages")
def get_messages(
    task_id: str,
    user_id: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    principal: dict[str, str] | None = Depends(get_principal),
) -> dict:
    try:
        service = get_service()
        context = service.load(task_id)
        bound_user_id = resolve_user_id(user_id, principal)
        if not session_id or context.user_id != bound_user_id or context.session_id != session_id:
            raise PermissionError("task does not belong to this session")
        return service.user_view(context)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="事项不存在。") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="无权访问该事项。") from exc
