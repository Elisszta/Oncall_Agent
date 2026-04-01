from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from services.search_engine import search_v2

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/search")
def search(q: str = ""):
    results = search_v2(q, use_rrf=True, vector_weight=0.8)
    return {
        "query": q,
        "results": results
    }

@router.get("/", response_class=HTMLResponse)
def get_search_page(request: Request):
    return templates.TemplateResponse(request=request, name="v2/index.html")
