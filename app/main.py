from fastapi import FastAPI
from app.routers import chat, rag

app = FastAPI(
    title="SmartHub AI",
    description="AI/LLM Engine for SmartHub - G4 Delta",
    version="1.0.0"
)

# Include routers
app.include_router(chat.router, tags=["Chat"])
app.include_router(rag.router, tags=["RAG"])

@app.get("/")
def home():
    return {"message": "SmartHub AI is running!"}

@app.get("/about")
def about():
    return {"project": "SmartHub", "role": "AI/LLM", "group": "G4 Delta"}