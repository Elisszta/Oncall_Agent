from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from services.document_store import store
from services.search_engine import search_v1

router = APIRouter()
templates = Jinja2Templates(directory="templates")

class DocumentCreate(BaseModel):
    id: str
    html: str

@router.post("/documents", status_code=201)
def add_document(doc: DocumentCreate):
    new_doc = store.add_document(doc.id, doc.html)
    return {"id": new_doc.id, "title": new_doc.title}

@router.get("/search")
def search(q: str = ""):
    results = search_v1(q)
    return {
        "query": q,
        "results": results
    }

@router.get("/", response_class=HTMLResponse)
def get_search_page(request: Request):
    return templates.TemplateResponse(request=request, name="v1/index.html")
