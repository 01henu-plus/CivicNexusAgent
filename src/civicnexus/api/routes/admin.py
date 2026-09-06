from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from civicnexus.api.auth import require_admin
from civicnexus.api.services import get_service

router = APIRouter()


class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


@router.post("/admin/login")
def login(request: LoginRequest) -> dict[str, str]:
    settings = get_service().settings
    if (
        request.username != settings.admin_username
        or request.password != settings.admin_password.get_secret_value()
    ):
        raise HTTPException(status_code=401, detail="管理员账号或密码错误。")
    return get_service().issue_admin_token()


@router.get("/admin/overview", dependencies=[Depends(require_admin)])
def overview() -> dict:
    return get_service().overview()


@router.get("/admin/tasks", dependencies=[Depends(require_admin)])
def tasks() -> dict:
    return {"tasks": get_service().task_list()}


@router.get("/admin/tasks/{task_id}", dependencies=[Depends(require_admin)])
def task_detail(task_id: str) -> dict:
    try:
        return get_service().admin_detail(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="事项不存在。") from exc


@router.get("/admin/metrics", dependencies=[Depends(require_admin)])
def metrics() -> dict:
    return get_service().metrics()


@router.get("/admin/evaluation/report", dependencies=[Depends(require_admin)])
def evaluation_report() -> dict:
    return get_service().evaluation_report()
