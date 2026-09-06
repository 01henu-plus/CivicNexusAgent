"""Minimal bearer authentication shared by user and admin routes."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException

from civicnexus.api.services import get_service


def _bearer(authorization: str | None) -> str:
    if not authorization:
        return ""
    scheme, _, token = authorization.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def get_principal(authorization: str | None = Header(default=None)) -> dict[str, str] | None:
    """Return the authenticated principal, or ``None`` for a guest request."""
    if not authorization:
        return None
    principal = get_service().resolve_token(_bearer(authorization))
    if principal is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录。")
    return principal


def require_admin(principal: dict[str, str] | None = Depends(get_principal)) -> dict[str, str]:
    if not principal or principal.get("role") != "admin":
        raise HTTPException(status_code=401, detail="需要管理员登录。")
    return principal


def resolve_user_id(requested: str | None, principal: dict[str, str] | None) -> str:
    """Bind a user token to its own namespace while retaining guest support."""
    if principal and principal.get("role") == "user":
        user_id = principal["user_id"]
        if requested and requested != user_id:
            raise HTTPException(status_code=403, detail="登录用户与请求身份不一致。")
        return user_id
    return requested or (principal or {}).get("user_id") or "anonymous"


__all__ = ["get_principal", "require_admin", "resolve_user_id"]
