import fitz  # PyMuPDF - reads PDFs
import chromadb
from sentence_transformers import SentenceTransformer
from docx import Document
import os

import pytesseract
from PIL import Image

# Tell pytesseract where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract OCR\tesseract.exe"

# Setup ChromaDB - persistent storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="smarthub_docs")

# Setup embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(file_path: str) -> str:
    """Read all text from a PDF file"""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Read all text from a DOCX file"""
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_txt(file_path: str) -> str:
    """Read all text from a TXT file"""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def extract_text_from_image(file_path: str) -> str:
    """Read text from an image using OCR"""
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return text

def get_document_text(filename: str) -> str:
    """
    Retrieve all chunks belonging to a document.
    """
    results = collection.get(
        where={"filename": filename}
    )

    documents = results.get("documents", [])

    return "\n".join(documents)

def extract_text(file_path: str, filename: str) -> str:
    """Decide which extractor to use based on file type"""
    extension = filename.split(".")[-1].lower()
    
    if extension == "pdf":
        return extract_text_from_pdf(file_path)
    elif extension == "docx":
        return extract_text_from_docx(file_path)
    elif extension == "txt":
        return extract_text_from_txt(file_path)
    elif extension in ["png", "jpg", "jpeg"]:
        return extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}. Only PDF, DOCX, TXT allowed.")

def chunk_text(text: str, chunk_size: int = 500) -> list:
    """Split text into smaller chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

import time

def ingest_document(file_path: str, filename: str) -> dict:
    """Process a document and store it in ChromaDB"""
    # Step 1: Extract text based on file type
    text = extract_text(file_path, filename)
    
    # Step 2: Split into chunks
    chunks = chunk_text(text)
    
    # Step 3: Create embeddings and store with upload timestamp
    upload_time = time.time()
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{filename}_chunk_{i}"],
            metadatas=[
                {
                    "filename": filename,
                    "uploaded_at": upload_time
                }
            ]
        )
    
    return {"filename": filename, "chunks_stored": len(chunks)}

def query_documents(question: str, filename: str = None, search_all: bool = False) -> str:
    """Search ChromaDB for relevant chunks.
    - If filename given: search only that file
    - If search_all=True: search across all documents
    - Otherwise (default): search only the most recently uploaded file
    """
    question_embedding = embedding_model.encode(question).tolist()
    
    if filename:
        target_file = filename
    elif search_all:
        target_file = None
    else:
        target_file = get_latest_document()
    
    if target_file:
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3,
            where={"filename": target_file}
        )
    else:
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )
    
    chunks = results["documents"][0]
    
    if not chunks:
        return ""
    
    return "\n\n".join(chunks)

def list_documents() -> list:
    """Get all unique filenames stored in ChromaDB"""
    results = collection.get()
    
    if not results["metadatas"]:
        return []
    
    # Extract unique filenames from metadata
    filenames = list(set(
        meta["filename"] 
        for meta in results["metadatas"]
        if "filename" in meta
    ))
    return filenames

def get_latest_document() -> str:
    """Get the filename of the most recently uploaded document"""
    results = collection.get()
    
    if not results["metadatas"]:
        return None
    
    # Find the metadata with the highest uploaded_at timestamp
    latest = max(results["metadatas"], key=lambda m: m.get("uploaded_at", 0))
    return latest.get("filename")

def delete_document(filename: str) -> dict:
    """Delete all chunks belonging to a filename"""
    results = collection.get(
        where={"filename": filename}
    )
    
    ids = results["ids"]
    
    if not ids:
        raise ValueError(f"Document '{filename}' not found in database")
    
    collection.delete(ids=ids)
    
    return {"deleted": filename, "chunks_removed": len(ids)}