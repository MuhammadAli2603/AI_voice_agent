"""
Whisper Speech-to-Text implementation using Hugging Face Inference API
No local model download required!
"""
import requests
import numpy as np
from typing import Union
import base64
import io

from app.modules.stt.base import BaseSTT
from app.utils.logger import log
from app.utils.audio_utils import load_audio, audio_to_bytes


class WhisperSTT(BaseSTT):
    """
    Whisper STT using Hugging Face Inference API
    Uses cloud-based inference - no local models needed!
    """
    
    def __init__(self, model_name: str = "openai/whisper-large-v3", api_key: str = ""):
        """
        Initialize Whisper STT with API
        
        Args:
            model_name: Whisper model ID on Hugging Face
            api_key: Hugging Face API key
        """
        super().__init__(model_name, device="api")
        self.api_key = api_key
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.load_model()
        
    def load_model(self):
        """Verify API access"""
        try:
            log.info(f"Initializing API connection for: {self.model_name}")
            
            if not self.api_key or self.api_key == "your_huggingface_api_key_here":
                log.warning("⚠️  No valid Hugging Face API key provided!")
                log.warning("Get your API key from: https://huggingface.co/settings/tokens")
                log.warning("Set it in .env file: HUGGINGFACE_API_KEY=your_key")
            
            # Mark as loaded (API connection will be tested on first request)
            self.model = "api"
            log.info("✓ STT API initialized successfully")
            
        except Exception as e:
            log.error(f"Error initializing STT API: {e}")
            raise
    
    def transcribe(
        self,
        audio: Union[np.ndarray, str, bytes],
        language: str = "en"
    ) -> dict:
        """
        Transcribe audio to text using Hugging Face API
        
        Args:
            audio: Audio array, file path, or bytes
            language: Language code
            
        Returns:
            Dictionary with 'text' and 'confidence'
        """
        try:
            # Convert audio to bytes
            if isinstance(audio, str):
                # Load from file
                audio_array, _ = load_audio(audio, sample_rate=16000)
                audio_bytes = audio_to_bytes(audio_array, sample_rate=16000)
            elif isinstance(audio, np.ndarray):
                # Convert numpy array to bytes
                audio_bytes = audio_to_bytes(audio, sample_rate=16000)
            elif isinstance(audio, bytes):
                audio_bytes = audio
            else:
                raise ValueError(f"Unsupported audio type: {type(audio)}")
            
            log.info(f"Sending audio to Hugging Face API ({len(audio_bytes)} bytes)")
            
            # Make API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                data=audio_bytes,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, dict) and "text" in result:
                    text = result["text"]
                elif isinstance(result, list) and len(result) > 0:
                    text = result[0].get("text", "")
                else:
                    text = str(result)
                
                log.info(f"✓ Transcription: {text}")
                
                return {
                    "text": text.strip(),
                    "confidence": None
                }
            
            elif response.status_code == 503:
                log.warning("Model is loading... This may take a minute on first use")
                try:
                    error_msg = response.json().get("error", "Model loading")
                except:
                    error_msg = "Model loading"
                return {
                    "text": f"[Model Loading: {error_msg}]",
                    "confidence": None
                }

            else:
                # Try to parse error message
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", response.text)
                except:
                    error_msg = response.text[:500]  # First 500 chars of response

                log.error(f"API Error ({response.status_code}): {error_msg}")
                raise Exception(f"API Error ({response.status_code}): {error_msg}")
            
        except requests.exceptions.Timeout:
            log.error("API request timed out")
            raise Exception("API request timed out. Try again.")
        except Exception as e:
            log.error(f"Error during transcription: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if API is initialized"""
        return self.model == "api"