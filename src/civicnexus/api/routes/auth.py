from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from civicnexus.api.auth import get_principal
from civicnexus.api.services import get_service

router = APIRouter()


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=128)
    password: str = Field(min_length=6, max_length=256)
    # The UI sends an empty string when the optional display name is omitted;
    # the service normalizes that value to the username.
    display_name: str | None = Field(default=None, max_length=128)


@router.post("/auth/login")
def login(request: LoginRequest) -> dict[str, str]:
    result = get_service().authenticate(request.username, request.password)
    if result is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误。")
    return result


@router.post("/auth/register")
def register(request: RegisterRequest) -> dict[str, str]:
    try:
        return get_service().register_user(
            request.username,
            request.password,
            request.display_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail="用户名已被占用。") from exc


@router.get("/auth/me")
def me(principal: dict[str, str] | None = Depends(get_principal)) -> dict[str, str]:
    if principal is None:
        raise HTTPException(status_code=401, detail="需要登录。")
    return principal
