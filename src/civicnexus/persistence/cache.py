"""Redis activity cache and deterministic local equivalent."""

from __future__ import annotations
import json
import threading
import time
from typing import Any, Callable
from uuid import UUID
from redis import Redis
from civicnexus.core.config import get_settings
from civicnexus.domain.context import TaskContext


def _key(task_id: UUID | str) -> str:
    return f"civicnexus:checkpoint:{task_id}"


def _snapshot_key(prefix: str, task_id: UUID | str) -> str:
    return f"{prefix}:snapshot:{task_id}"


def _dump(value: Any) -> str:
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, default=str)


def _load(raw: str | bytes | None) -> Any:
    if raw is None:
        return None
    value = json.loads(raw)
    model = (
        TaskContext
        if isinstance(value, dict) and {"user_message", "status"} <= value.keys()
        else None
    )
    if (
        model is None
        and isinstance(value, dict)
        and {"rendered_context", "task_id"} <= value.keys()
    ):
        from civicnexus.memory.models import ContextSnapshot

        model = ContextSnapshot
    if model is not None:
        try:
            return model.model_validate(value)
        except ValueError:
            pass
    return value


def _args(first: Any, second: Any | None) -> tuple[Any, UUID | str]:
    if second is None:
        value, task_id = first, getattr(first, "task_id", None)
    else:
        task_id, value = first, second
    if task_id is None:
        raise ValueError("task_id is required")
    return value, task_id


def _ttl(seconds: int, override: int | None) -> int:
    return seconds if override is None else override


class CheckpointStore:
    def save_context(
        self,
        task_id: UUID | str | TaskContext,
        context: Any | None = None,
        *,
        ttl_seconds: int = 86400,
        ttl: int | None = None,
    ) -> None:
        raise NotImplementedError

    def get_context(self, task_id: UUID | str) -> Any | None:
        raise NotImplementedError

    def delete_context(self, task_id: UUID | str) -> None:
        raise NotImplementedError

    def save_checkpoint(
        self,
        task_id: UUID | str | TaskContext,
        context: Any | None = None,
        *,
        ttl_seconds: int = 86400,
        ttl: int | None = None,
    ) -> None:
        self.save_context(task_id, context, ttl_seconds=ttl_seconds, ttl=ttl)

    def get_checkpoint(self, task_id: UUID | str) -> Any | None:
        return self.get_context(task_id)

    def delete_checkpoint(self, task_id: UUID | str) -> None:
        self.delete_context(task_id)

    def save_snapshot(
        self,
        task_id: Any,
        snapshot: Any | None = None,
        *,
        ttl_seconds: int = 86400,
        ttl: int | None = None,
    ) -> None:
        value, task_id = _args(task_id, snapshot)
        self.save_context(task_id, value, ttl_seconds=ttl_seconds, ttl=ttl)

    def get_snapshot(self, task_id: UUID | str) -> Any | None:
        return self.get_context(task_id)


class RedisCheckpointStore(CheckpointStore):
    def __init__(
        self,
        redis_url: str | None = None,
        *,
        client: Redis | None = None,
        prefix: str = "civicnexus:checkpoint",
    ) -> None:
        self.client = client or Redis.from_url(
            redis_url or get_settings().redis_url, decode_responses=False
        )
        self.prefix = prefix.rstrip(":")

    def _key(self, task_id: UUID | str) -> str:
        return f"{self.prefix}:{task_id}"

    def save_context(
        self,
        task_id: UUID | str | TaskContext,
        context: Any | None = None,
        *,
        ttl_seconds: int = 86400,
        ttl: int | None = None,
    ) -> None:
        value, task_id = _args(task_id, context)
        self.client.set(self._key(task_id), _dump(value), ex=_ttl(ttl_seconds, ttl))

    def get_context(self, task_id: UUID | str) -> Any | None:
        return _load(self.client.get(self._key(task_id)))

    def delete_context(self, task_id: UUID | str) -> None:
        self.client.delete(self._key(task_id), _snapshot_key(self.prefix, task_id))

    def save_snapshot(
        self,
        task_id: Any,
        snapshot: Any | None = None,
        *,
        ttl_seconds: int = 86400,
        ttl: int | None = None,
    ) -> None:
        value, task_id = _args(task_id, snapshot)
        self.client.set(
            _snapshot_key(self.prefix, task_id), _dump(value), ex=_ttl(ttl_seconds, ttl)
        )

    def get_snapshot(self, task_id: UUID | str) -> Any | None:
        return _load(self.client.get(_snapshot_key(self.prefix, task_id)))

    def ping(self) -> bool:
        return bool(self.client.ping())

    def close(self) -> None:
        self.client.close()


class MemoryCache(CheckpointStore):
    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock, self._values = clock, {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(task_id: UUID | str) -> str:
        return _key(task_id)

    @staticmethod
    def _snapshot_key(task_id: UUID | str) -> str:
        return _snapshot_key("civicnexus:checkpoint", task_id)

    def _put(self, key: str, value: Any, seconds: int) -> None:
        expiry = None if seconds <= 0 else self._clock() + seconds
        self._values[key] = (expiry, _dump(value))

    def _get(self, key: str) -> Any | None:
        item = self._values.get(key)
        if item is None:
            return None
        expiry, raw = item
        if expiry is not None and self._clock() >= expiry:
            self._values.pop(key, None)
            return None
        return _load(raw)

    def save_context(
        self,
        task_id: UUID | str | TaskContext,
        context: Any | None = None,
        *,
        ttl_seconds: int = 86400,
        ttl: int | None = None,
    ) -> None:
        value, task_id = _args(task_id, context)
        with self._lock:
            self._put(_key(task_id), value, _ttl(ttl_seconds, ttl))

    def get_context(self, task_id: UUID | str) -> Any | None:
        with self._lock:
            return self._get(_key(task_id))

    def delete_context(self, task_id: UUID | str) -> None:
        with self._lock:
            self._values.pop(_key(task_id), None)
            self._values.pop(self._snapshot_key(task_id), None)

    def save_snapshot(
        self,
        task_id: Any,
        snapshot: Any | None = None,
        *,
        ttl_seconds: int = 86400,
        ttl: int | None = None,
    ) -> None:
        value, task_id = _args(task_id, snapshot)
        with self._lock:
            self._put(self._snapshot_key(task_id), value, _ttl(ttl_seconds, ttl))

    def get_snapshot(self, task_id: UUID | str) -> Any | None:
        with self._lock:
            return self._get(self._snapshot_key(task_id))

    def clear(self) -> None:
        with self._lock:
            self._values.clear()


__all__ = ["CheckpointStore", "RedisCheckpointStore", "MemoryCache"]
