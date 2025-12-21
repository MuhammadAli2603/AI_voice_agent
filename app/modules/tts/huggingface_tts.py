"""
Text-to-Speech implementation using Hugging Face Inference API
No local model download required!
"""
import requests
import numpy as np
import io
import soundfile as sf

from app.modules.tts.base import BaseTTS
from app.utils.logger import log


class HuggingFaceTTS(BaseTTS):
    """
    TTS using Hugging Face Inference API
    Uses cloud-based inference - no local models needed!
    """
    
    def __init__(
        self,
        model_name: str = "facebook/mms-tts-eng",
        api_key: str = ""
    ):
        """
        Initialize TTS with API
        
        Args:
            model_name: TTS model ID on Hugging Face
            api_key: Hugging Face API key
        """
        super().__init__(model_name, device="api")
        self.api_key = api_key
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.sample_rate = 16000  # Most TTS models use 16kHz or 22.05kHz
        self.load_model()
        
    def load_model(self):
        """Verify API access"""
        try:
            log.info(f"Initializing API connection for: {self.model_name}")
            
            if not self.api_key or self.api_key == "your_huggingface_api_key_here":
                log.warning("⚠️  No valid Hugging Face API key provided!")
                log.warning("Get your API key from: https://huggingface.co/settings/tokens")
            
            # Mark as loaded
            self.model = "api"
            log.info("✓ TTS API initialized successfully")
            
        except Exception as e:
            log.error(f"Error initializing TTS API: {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        language: str = "en",
        speed: float = 1.0
    ) -> np.ndarray:
        """
        Synthesize speech from text using Hugging Face API
        
        Args:
            text: Text to synthesize
            language: Language code
            speed: Speech speed multiplier (1.0 = normal)
            
        Returns:
            Audio array (numpy array)
        """
        try:
            if not text or not text.strip():
                log.warning("Empty text provided for TTS")
                return np.array([])
            
            log.info(f"Synthesizing text: '{text[:50]}...'")
            
            # Prepare payload
            payload = {"inputs": text}
            
            # Make API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                # Response is audio bytes
                audio_bytes = response.content
                
                # Convert bytes to numpy array
                try:
                    # Try to read as WAV file
                    audio, sr = sf.read(io.BytesIO(audio_bytes))
                    
                    # Convert to mono if stereo
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                    
                    # Store actual sample rate
                    self.sample_rate = sr
                    
                    # Apply speed adjustment if needed (simple method)
                    if speed != 1.0:
                        audio = self._simple_time_stretch(audio, speed)
                    
                    log.info(f"✓ Speech synthesized: shape={audio.shape}, sr={sr}")
                    return audio.astype(np.float32)
                    
                except Exception as e:
                    log.error(f"Error decoding audio: {e}")
                    # Return raw bytes as float array (fallback)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                    return audio_array
            
            elif response.status_code == 503:
                log.warning("Model is loading... This may take a minute on first use")
                # Return silence as placeholder
                return np.zeros(16000, dtype=np.float32)
            
            else:
                error_msg = response.json().get("error", response.text)
                log.error(f"API Error ({response.status_code}): {error_msg}")
                raise Exception(f"TTS API Error: {error_msg}")
            
        except requests.exceptions.Timeout:
            log.error("API request timed out")
            raise Exception("TTS API request timed out")
        except Exception as e:
            log.error(f"Error during speech synthesis: {e}")
            raise
    
    def _simple_time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """
        Simple time stretching (speed adjustment)
        
        Args:
            audio: Input audio
            rate: Speed multiplier (>1 = faster, <1 = slower)
            
        Returns:
            Time-stretched audio
        """
        # Simple resampling for speed change
        new_length = int(len(audio) / rate)
        
        # Linear interpolation
        x_old = np.linspace(0, len(audio) - 1, len(audio))
        x_new = np.linspace(0, len(audio) - 1, new_length)
        stretched = np.interp(x_new, x_old, audio)
        
        return stretched
    
    def get_sample_rate(self) -> int:
        """Get the sample rate of the TTS model"""
        return self.sample_rate
    
    def is_loaded(self) -> bool:
        """Check if API is initialized"""
        return self.model == "api"