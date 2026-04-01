import json
import os
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from services.agent import stream_chat

router = APIRouter()
templates = Jinja2Templates(directory="templates")

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json")

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "use_embedding_search": True
        }
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(config_data):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

@router.get("/", response_class=HTMLResponse)
async def get_v3_page(request: Request):
    config = load_config()
    return templates.TemplateResponse(
        request=request, 
        name="v3/index.html", 
        context={"config": config}
    )

class ChatMessage(BaseModel):
    role: str
    content: str
    
class ChatRequest(BaseModel):
    messages: list[ChatMessage]

@router.post("/config")
async def update_config(config: dict):
    save_config(config)
    return {"status": "success"}

@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    config = load_config()
    messages_list = [{"role": msg.role, "content": msg.content} for msg in req.messages]
    return StreamingResponse(
        stream_chat(messages_list, config),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

