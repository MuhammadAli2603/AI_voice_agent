"""
Configuration management for AI Voice Agent
Python 3.14 compatible, no Pydantic required
Supports Hugging Face API-based models
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class Settings:
    """Application settings for API-based models"""

    def __init__(self):
        # Server
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8000"))
        self.debug: bool = os.getenv("DEBUG", "True").lower() == "true"

        # Hugging Face API
        self.huggingface_api_key: str = os.getenv("HUGGINGFACE_API_KEY", "")

        # Models (Hugging Face Model IDs)
        self.stt_model: str = os.getenv("STT_MODEL", "openai/whisper-large-v3")
        self.llm_model: str = os.getenv("LLM_MODEL", "microsoft/DialoGPT-medium")
        self.tts_model: str = os.getenv("TTS_MODEL", "facebook/mms-tts-eng")

        # Device (CPU or GPU)
        self.device: str = os.getenv("DEVICE", "cpu")  # "cpu" or "cuda"

        # Audio settings
        self.sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
        self.max_audio_length: int = int(os.getenv("MAX_AUDIO_LENGTH", "30"))

        # LLM settings
        self.llm_max_length: int = int(os.getenv("LLM_MAX_LENGTH", "512"))
        self.llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.llm_top_p: float = float(os.getenv("LLM_TOP_P", "0.9"))
        self.llm_max_new_tokens: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "150"))

        # API settings
        self.api_timeout: int = int(os.getenv("API_TIMEOUT", "30"))

        # Logging
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def model_dump(self) -> dict:
        """Return settings as a dictionary"""
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "stt_model": self.stt_model,
            "llm_model": self.llm_model,
            "tts_model": self.tts_model,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "max_audio_length": self.max_audio_length,
            "llm_max_length": self.llm_max_length,
            "llm_temperature": self.llm_temperature,
            "llm_top_p": self.llm_top_p,
            "llm_max_new_tokens": self.llm_max_new_tokens,
            "api_timeout": self.api_timeout,
            "log_level": self.log_level,
        }


# Global settings instance
settings = Settings()
