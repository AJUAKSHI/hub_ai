from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.services.rag_service import ingest_document, query_documents
from app.services.ai_service import get_ai_response
import os
import shutil

from app.services.ai_service import summarize_text
from app.services.rag_service import get_document_text

from app.services.rag_service import collection

router = APIRouter()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file types
ALLOWED_EXTENSIONS = ["pdf", "docx", "txt", "png", "jpg", "jpeg"]


@router.post("/rag/ingest")
def ingest(file: UploadFile = File(...)):
    # Check file type first
    extension = file.filename.split(".")[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{extension}' not supported. Only PDF, DOCX, TXT allowed."
        )
    
    # Save the uploaded file
    file_path = f"{UPLOAD_FOLDER}/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process and store in ChromaDB
    try:
        result = ingest_document(file_path, file.filename)
        return {
            "message": "Document uploaded and processed successfully!",
            "filename": result["filename"],
            "chunks_stored": result["chunks_stored"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class QueryRequest(BaseModel):
    question: str
    filename: str = None
    search_all: bool = False

@router.post("/rag/query")
def query(request: QueryRequest):
    try:
        # Search ChromaDB for relevant chunks (filtered if filename given)
        relevant_chunks = query_documents(request.question, request.filename, request.search_all)
        
        if not relevant_chunks:
            raise HTTPException(
                status_code=404,
                detail=f"No content found for '{request.filename}'. Check the filename is correct."
            )
        
        # Build prompt with context
        prompt = f"""You are SmartHub AI. Answer the question using the context below.
        
Context from uploaded documents:
{relevant_chunks}

Question: {request.question}

Answer:"""
        
        result = get_ai_response(prompt)
        return {
            "question": request.question,
            "filename": request.filename if request.filename else "All documents",
            "answer": result["reply"],
            "source": "SmartHub RAG Pipeline",
            "token_usage": result["token_usage"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rag/documents")
def list_documents():
    """Get all uploaded documents"""
    try:
        from app.services.rag_service import list_documents
        files = list_documents()
        return {
            "documents": files,
            "total": len(files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/rag/documents/{filename}")
def delete_document(filename: str):
    """Delete a specific document from ChromaDB"""
    try:
        from app.services.rag_service import delete_document
        result = delete_document(filename)
        return {
            "message": f"Document deleted successfully!",
            "filename": result["deleted"],
            "chunks_removed": result["chunks_removed"]
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SummaryRequest(BaseModel):
    filename: str

@router.post("/rag/summarize")
def summarize(request: SummaryRequest):

    text = get_document_text(request.filename)

    if not text:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    summary = summarize_text(text)

    return {
        "filename": request.filename,
        "summary": summary
    }

