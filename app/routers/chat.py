from fastapi import APIRouter
from pydantic import BaseModel
from app.services.ai_service import get_ai_response
from typing import List

router = APIRouter()

class Message(BaseModel):
    role: str # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message]=[]  #optional, empty by default

@router.post("/chat")
def chat(request: ChatRequest):
    result = get_ai_response(request.message, request.history)
    return {
        "you_said": request.message,
        "reply": result["reply"],
        "token_usage": result["token_usage"]
    }

from fastapi.responses import StreamingResponse
import json
from app.services.ai_service import chat_stream

@router.post("/api/v1/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Stream AI response as Server-Sent Events"""
    
    async def event_generator():
        full_response = ""
        async for token in chat_stream(request.history + [{"role": "user", "content": request.message}]):
            full_response += token
            yield f"data: {json.dumps({'delta': token})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")