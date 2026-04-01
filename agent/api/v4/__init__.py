from fastapi import APIRouter
from .router import router as v4_router

router = APIRouter()
router.include_router(v4_router)
