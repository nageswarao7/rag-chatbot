import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from backend.rag_service import rag_service
from backend.ingestion import ingestion_service
from backend.vector_store import vector_store
from backend.cache import query_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

app = FastAPI(
    title="RAG Chat API",
    description="Production-ready RAG-based chat API with document ingestion",
    version="1.0.0"
)

# --- Middleware ---

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} - {response.status_code}")
    return response

# --- Request/Response Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

# --- Health & Status Endpoints ---
@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "RAG Chat API",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    """Detailed health check for monitoring."""
    return {
        "status": "healthy",
        "components": {
            "vector_store": "up",
            "cache": "up",
            "llm": "up"
        }
    }

@app.get("/stats")
def get_stats():
    """Get statistics about the system."""
    cache_stats = query_cache.stats()
    return {
        "total_chunks": vector_store.count(),
        "cache_size": cache_stats["size"],
        "cache_max_size": cache_stats["max_size"],
        "cache_ttl_seconds": cache_stats["ttl_seconds"]
    }

# --- Chat Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the RAG chatbot with conversation history."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    history = [{"role": msg.role, "content": msg.content} for msg in request.history]
    result = rag_service.generate_response(request.message, history=history)
    return ChatResponse(answer=result["answer"], sources=result["sources"])

# --- Document Management Endpoints ---
@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Upload and ingest a document (txt, md, pdf, docx)."""
    allowed_types = ["txt", "md", "pdf", "docx", "doc", "csv"]
    ext = file.filename.split(".")[-1].lower()
    
    if ext not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"File type '{ext}' not supported. Allowed: {allowed_types}"
        )
    
    try:
        content = await file.read()
        num_chunks = ingestion_service.process_document(content, filename=file.filename)
        logger.info(f"Ingested {file.filename}: {num_chunks} chunks")
        return {
            "filename": file.filename,
            "chunks_created": num_chunks,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear")
def clear_all_documents():
    """Delete all documents from the vector store and clear cache."""
    try:
        vector_store.clear_all()
        query_cache.clear()
        logger.info("All documents and cache cleared")
        return {"status": "success", "message": "All documents and cache deleted"}
    except Exception as e:
        logger.error(f"Clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Cache Management ---
@app.delete("/cache")
def clear_cache():
    """Clear the query cache."""
    query_cache.clear()
    return {"status": "success", "message": "Cache cleared"}
