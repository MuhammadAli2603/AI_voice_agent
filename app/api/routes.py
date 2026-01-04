"""
FastAPI routes for Voice Agent API
Python 3.14 compatible - no Pydantic v2!
"""
import base64
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional, Dict, Any

from app.pipeline.voice_pipeline import VoiceAgentPipeline
from app.utils.logger import log

# Initialize router
router = APIRouter()

# Initialize pipeline (will be set in main.py)
pipeline: Optional[VoiceAgentPipeline] = None


def set_pipeline(voice_pipeline: VoiceAgentPipeline):
    """Set the pipeline instance"""
    global pipeline
    pipeline = voice_pipeline


@router.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "AI Voice Agent API",
        "version": "1.0.0",
        "status": "running",
        "python": "3.14 compatible"
    }


@router.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    health = pipeline.health_check()
    
    return {
        "status": "healthy" if health["pipeline"] else "degraded",
        "modules": health
    }


@router.post("/transcribe", tags=["STT"])
async def transcribe_audio(request: Dict[str, Any]):
    """
    Transcribe audio to text (STT only)
    
    Body: {"audio_base64": "...", "language": "en"}
    """
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        audio_base64 = request.get("audio_base64")
        language = request.get("language", "en")
        
        if not audio_base64:
            raise HTTPException(status_code=400, detail="audio_base64 is required")
        
        # Decode audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Transcribe
        result = pipeline.transcribe_only(
            audio_input=audio_bytes,
            language=language
        )
        
        return {
            "text": result["text"],
            "confidence": result.get("confidence"),
            "processing_time": result["processing_time"]
        }
        
    except Exception as e:
        log.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe/file", tags=["STT"])
async def transcribe_audio_file(
    file: UploadFile = File(...),
    language: str = "en"
):
    """
    Transcribe audio file to text (STT only)
    """
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        # Read file
        audio_bytes = await file.read()
        
        # Transcribe
        result = pipeline.transcribe_only(
            audio_input=audio_bytes,
            language=language
        )
        
        return {
            "text": result["text"],
            "confidence": result.get("confidence"),
            "processing_time": result["processing_time"]
        }
        
    except Exception as e:
        log.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", tags=["LLM"])
async def chat(request: Dict[str, Any]):
    """
    Generate chat response (LLM only) with optional KB integration

    Body: {"message": "...", "conversation_history": [...], "company_id": "techstore"}
    """
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        message = request.get("message")
        if not message:
            raise HTTPException(status_code=400, detail="message is required")

        conversation_history = request.get("conversation_history")
        company_id = request.get("company_id")

        result = pipeline.generate_response_only(
            message=message,
            conversation_history=conversation_history,
            company_id=company_id
        )

        return {
            "response": result["response"],
            "processing_time": result["processing_time"],
            "kb_enabled": company_id is not None
        }

    except Exception as e:
        log.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthesize", tags=["TTS"])
async def synthesize_speech(request: Dict[str, Any]):
    """
    Synthesize text to speech (TTS only)
    
    Body: {"text": "...", "language": "en", "speed": 1.0}
    """
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        text = request.get("text")
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        
        language = request.get("language", "en")
        speed = request.get("speed", 1.0)
        
        result = pipeline.synthesize_only(
            text=text,
            language=language,
            speed=speed
        )
        
        return {
            "audio_base64": result["audio_base64"],
            "processing_time": result["processing_time"]
        }
        
    except Exception as e:
        log.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice-agent", tags=["Voice Agent"])
async def voice_agent(request: Dict[str, Any]):
    """
    Complete voice agent pipeline with optional KB integration

    Body: {
        "audio_base64": "...",
        "language": "en",
        "conversation_history": [...],
        "company_id": "techstore"
    }
    """
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        audio_base64 = request.get("audio_base64")
        if not audio_base64:
            raise HTTPException(status_code=400, detail="audio_base64 is required")

        language = request.get("language", "en")
        conversation_history = request.get("conversation_history")
        company_id = request.get("company_id")

        # Decode audio
        audio_bytes = base64.b64decode(audio_base64)

        # Process through pipeline
        result = pipeline.process_audio(
            audio_input=audio_bytes,
            language=language,
            conversation_history=conversation_history,
            company_id=company_id
        )

        return {
            "transcription": result["transcription"],
            "llm_response": result["llm_response"],
            "audio_base64": result["audio_base64"],
            "total_processing_time": result["timing"]["total"],
            "breakdown": {
                "stt_time": result["timing"]["stt"],
                "llm_time": result["timing"]["llm"],
                "tts_time": result["timing"]["tts"]
            },
            "kb_enabled": company_id is not None
        }

    except Exception as e:
        log.error(f"Voice agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice-agent/file", tags=["Voice Agent"])
async def voice_agent_file(
    file: UploadFile = File(...),
    language: str = "en",
    company_id: str = None
):
    """
    Complete voice agent pipeline with file upload and optional KB integration
    """
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        # Read file
        audio_bytes = await file.read()

        # Process through pipeline
        result = pipeline.process_audio(
            audio_input=audio_bytes,
            language=language,
            conversation_history=None,
            company_id=company_id
        )

        return {
            "transcription": result["transcription"],
            "llm_response": result["llm_response"],
            "audio_base64": result["audio_base64"],
            "total_processing_time": result["timing"]["total"],
            "breakdown": {
                "stt_time": result["timing"]["stt"],
                "llm_time": result["timing"]["llm"],
                "tts_time": result["timing"]["tts"]
            },
            "kb_enabled": company_id is not None
        }

    except Exception as e:
        log.error(f"Voice agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset", tags=["General"])
async def reset_conversation():
    """Reset conversation memory"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        pipeline.reset_conversation()
        return {"message": "Conversation reset successfully"}
        
    except Exception as e:
        log.error(f"Reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))