from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from services.document_store import load_initial_data
from api.v1.router import router as v1_router
from api.v2.router import router as v2_router
from api.v3.router import router as v3_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load initial documents from data/
    load_initial_data("data")
    yield


app = FastAPI(lifespan=lifespan, title="On-Call Assistant")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

app.mount("/data", StaticFiles(directory="data", html=True), name="data")
app.include_router(v1_router, prefix="/v1")
app.include_router(v2_router, prefix="/v2")
app.include_router(v3_router, prefix="/v3")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
