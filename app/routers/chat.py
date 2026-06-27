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