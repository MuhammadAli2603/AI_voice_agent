"""
Request/Response models using Python dataclasses
No Pydantic needed!
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class TranscriptionRequest:
    """Request model for STT"""
    audio_base64: Optional[str] = None
    language: str = "en"


@dataclass
class TranscriptionResponse:
    """Response model for STT"""
    text: str
    confidence: Optional[float] = None
    processing_time: float = 0.0


@dataclass
class ChatRequest:
    """Request model for LLM chat"""
    message: str
    conversation_history: Optional[List[dict]] = None


@dataclass
class ChatResponse:
    """Response model for LLM chat"""
    response: str
    processing_time: float = 0.0


@dataclass
class TTSRequest:
    """Request model for TTS"""
    text: str
    language: str = "en"
    speed: float = 1.0


@dataclass
class TTSResponse:
    """Response model for TTS"""
    audio_base64: str
    processing_time: float = 0.0


@dataclass
class VoiceAgentRequest:
    """Request model for complete voice agent pipeline"""
    audio_base64: str
    conversation_history: Optional[List[dict]] = None
    language: str = "en"


@dataclass
class VoiceAgentResponse:
    """Response model for complete voice agent pipeline"""
    transcription: str
    llm_response: str
    audio_base64: str
    total_processing_time: float
    breakdown: dict = field(default_factory=dict)


@dataclass
class HealthResponse:
    """Health check response"""
    status: str
    modules: dict = field(default_factory=dict)


# Helper function to convert dataclass to dict
def to_dict(obj):
    """Convert dataclass to dictionary"""
    if hasattr(obj, '__dataclass_fields__'):
        return {
            k: to_dict(v) for k, v in obj.__dict__.items()
        }
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_dict(item) for item in obj]
    else:
        return obj