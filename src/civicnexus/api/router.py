from fastapi import APIRouter

from civicnexus.api.routes.cases import router as cases_router
from civicnexus.api.routes.health import router as health_router
from civicnexus.api.routes.tasks import router as tasks_router
from civicnexus.api.routes.admin import router as admin_router
from civicnexus.api.routes.auth import router as auth_router


api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(cases_router, tags=["cases"])
api_router.include_router(tasks_router, tags=["tasks"])
api_router.include_router(admin_router, tags=["admin"])
api_router.include_router(auth_router, tags=["auth"])
