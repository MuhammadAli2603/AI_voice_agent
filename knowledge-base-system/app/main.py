"""
Main FastAPI application for the Knowledge Base System.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time

from app.api.endpoints import router
from app.config import settings
from app.utils.logger import setup_logging, get_logger, set_request_id
from app import __version__

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="RAG-based Knowledge Base Management System for AI Voice Agents",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to each request for tracing."""
    request_id = set_request_id()
    request.state.request_id = request_id

    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)

    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Time: {process_time:.3f}s"
    )

    return response


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid value",
            "detail": str(exc)
        }
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    """Handle FileNotFoundError exceptions."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "File not found",
            "detail": str(exc)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An error occurred"
        }
    )


# Include API routes
app.include_router(router, prefix="/api/v1")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with system information."""
    return {
        "name": settings.app_name,
        "version": __version__,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info(f"Starting {settings.app_name} v{__version__}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Embedding model: {settings.embedding_model_name}")
    logger.info(f"Vector DB: {settings.vector_db_type}")
    logger.info("Application started successfully")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down application")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
