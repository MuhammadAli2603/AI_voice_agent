"""
FastAPI application entry point
"""
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.utils.logger import log
from app.api import routes
from app.api.websocket import WebSocketHandler
from app.pipeline.voice_pipeline import VoiceAgentPipeline


# Global pipeline instance
pipeline = None
websocket_handler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    global pipeline, websocket_handler
    
    log.info("Starting AI Voice Agent...")
    log.info(f"Configuration: {settings.model_dump()}")
    
    try:
        # Initialize pipeline
        pipeline = VoiceAgentPipeline(
            stt_model=settings.stt_model,
            llm_model=settings.llm_model,
            tts_model=settings.tts_model,
            api_key=settings.huggingface_api_key
        )
        
        # Set pipeline in routes
        routes.set_pipeline(pipeline)
        
        # Initialize WebSocket handler
        websocket_handler = WebSocketHandler(pipeline)
        
        log.info("AI Voice Agent started successfully!")
        
        yield
        
    except Exception as e:
        log.error(f"Failed to start application: {e}")
        raise
    
    finally:
        # Shutdown
        log.info("Shutting down AI Voice Agent...")


# Create FastAPI app
app = FastAPI(
    title="AI Voice Agent",
    description="Complete open-source voice agent with STT, LLM, and TTS modules",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(routes.router)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice streaming
    """
    if websocket_handler:
        await websocket_handler.handle_voice_stream(websocket)
    else:
        await websocket.close(code=1011, reason="Service unavailable")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )