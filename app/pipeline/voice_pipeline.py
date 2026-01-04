"""
Voice Agent Pipeline - API-based (No local models!)
Orchestrates STT -> LLM -> TTS using Hugging Face Inference API
"""
import time
import base64
import io
from typing import Dict, Optional, List
import numpy as np

from app.modules.stt.whisper_stt import WhisperSTT
from app.modules.llm.receptionist_llm import ReceptionistLLM
from app.modules.tts.huggingface_tts import HuggingFaceTTS
from app.utils.logger import log
from app.utils.audio_utils import load_audio, audio_to_bytes
from app.config import settings


class VoiceAgentPipeline:
    """
    Complete voice agent pipeline using Hugging Face Inference API
    Processes: Audio Input -> Transcription -> LLM Response -> Speech Output
    """
    
    def __init__(
        self,
        stt_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        tts_model: Optional[str] = None,
        api_key: Optional[str] = None,
        kb_service_url: Optional[str] = None
    ):
        """
        Initialize voice agent pipeline with API and KB integration

        Args:
            stt_model: STT model ID (defaults to config)
            llm_model: LLM model ID (defaults to config)
            tts_model: TTS model ID (defaults to config)
            api_key: Hugging Face API key (defaults to config)
            kb_service_url: Knowledge Base service URL (defaults to config)
        """
        self.stt_model_name = stt_model or settings.stt_model
        self.llm_model_name = llm_model or settings.llm_model
        self.tts_model_name = tts_model or settings.tts_model
        self.api_key = api_key or settings.huggingface_api_key
        self.kb_service_url = kb_service_url or getattr(settings, 'kb_service_url', 'http://localhost:8001')
        
        # Initialize modules
        log.info("Initializing Voice Agent Pipeline with API...")
        self.stt = None
        self.llm = None
        self.tts = None
        
        self._initialize_modules()
        
    def _initialize_modules(self):
        """Initialize all API-based modules"""
        try:
            # Initialize STT
            log.info("Loading STT module (API)...")
            self.stt = WhisperSTT(
                model_name=self.stt_model_name,
                api_key=self.api_key
            )
            
            # Initialize LLM
            log.info("Loading LLM module (API)...")
            self.llm = ReceptionistLLM(
                model_name=self.llm_model_name,
                api_key=self.api_key,
                max_length=settings.llm_max_length,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                kb_service_url=self.kb_service_url
            )
            
            # Initialize TTS
            log.info("Loading TTS module (API)...")
            self.tts = HuggingFaceTTS(
                model_name=self.tts_model_name,
                api_key=self.api_key
            )
            
            log.info("✅ Voice Agent Pipeline initialized successfully!")
            log.info(f"   STT: {self.stt_model_name}")
            log.info(f"   LLM: {self.llm_model_name}")
            log.info(f"   TTS: {self.tts_model_name}")
            
        except Exception as e:
            log.error(f"Error initializing pipeline: {e}")
            raise
    
    def process_audio(
        self,
        audio_input: bytes,
        language: str = "en",
        conversation_history: Optional[List[Dict]] = None,
        company_id: Optional[str] = None
    ) -> Dict:
        """
        Process audio through complete pipeline with KB integration

        Args:
            audio_input: Audio bytes
            language: Language code
            conversation_history: Previous conversation
            company_id: Company ID for knowledge base context (optional)

        Returns:
            Dictionary with transcription, response, audio, and timing info
        """
        try:
            timing = {}
            
            # Step 1: Speech-to-Text
            log.info("Step 1: Transcribing audio...")
            start_time = time.time()
            
            audio, sr = load_audio(audio_input, sample_rate=16000)
            transcription_result = self.stt.transcribe(audio, language=language)
            transcription = transcription_result["text"]
            
            timing["stt"] = time.time() - start_time
            log.info(f"✓ Transcription: '{transcription}' (took {timing['stt']:.2f}s)")
            
            # Step 2: LLM Processing
            log.info("Step 2: Generating LLM response...")
            start_time = time.time()

            llm_response = self.llm.generate_response(
                message=transcription,
                conversation_history=conversation_history,
                company_id=company_id
            )

            timing["llm"] = time.time() - start_time
            log.info(f"✓ LLM Response: '{llm_response}' (took {timing['llm']:.2f}s)")
            
            # Step 3: Text-to-Speech
            log.info("Step 3: Synthesizing speech...")
            start_time = time.time()
            
            audio_output = self.tts.synthesize(
                text=llm_response,
                language=language
            )
            
            timing["tts"] = time.time() - start_time
            log.info(f"✓ Speech synthesized (took {timing['tts']:.2f}s)")
            
            # Convert audio to base64
            audio_bytes = audio_to_bytes(
                audio_output,
                sample_rate=self.tts.get_sample_rate()
            )
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Calculate total time
            timing["total"] = sum(timing.values())
            
            return {
                "transcription": transcription,
                "llm_response": llm_response,
                "audio_base64": audio_base64,
                "timing": timing
            }
            
        except Exception as e:
            log.error(f"Error in pipeline processing: {e}")
            raise
    
    def transcribe_only(self, audio_input: bytes, language: str = "en") -> Dict:
        """
        Only transcribe audio (STT only)
        
        Args:
            audio_input: Audio bytes
            language: Language code
            
        Returns:
            Transcription result
        """
        try:
            start_time = time.time()
            audio, sr = load_audio(audio_input, sample_rate=16000)
            result = self.stt.transcribe(audio, language=language)
            result["processing_time"] = time.time() - start_time
            return result
        except Exception as e:
            log.error(f"Error in transcription: {e}")
            raise
    
    def generate_response_only(
        self,
        message: str,
        conversation_history: Optional[List[Dict]] = None,
        company_id: Optional[str] = None
    ) -> Dict:
        """
        Only generate LLM response (LLM only) with KB integration

        Args:
            message: User message
            conversation_history: Previous conversation
            company_id: Company ID for knowledge base context (optional)

        Returns:
            LLM response
        """
        try:
            start_time = time.time()
            response = self.llm.generate_response(
                message=message,
                conversation_history=conversation_history,
                company_id=company_id
            )
            return {
                "response": response,
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            log.error(f"Error generating response: {e}")
            raise
    
    def synthesize_only(
        self,
        text: str,
        language: str = "en",
        speed: float = 1.0
    ) -> Dict:
        """
        Only synthesize speech (TTS only)
        
        Args:
            text: Text to synthesize
            language: Language code
            speed: Speech speed
            
        Returns:
            Audio data
        """
        try:
            start_time = time.time()
            audio = self.tts.synthesize(text, language=language, speed=speed)
            
            audio_bytes = audio_to_bytes(
                audio,
                sample_rate=self.tts.get_sample_rate()
            )
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            return {
                "audio_base64": audio_base64,
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            log.error(f"Error synthesizing speech: {e}")
            raise
    
    def reset_conversation(self):
        """Reset conversation memory"""
        if self.llm:
            self.llm.reset_conversation()
            log.info("Conversation reset")
    
    def health_check(self) -> Dict:
        """
        Check health of all modules
        
        Returns:
            Health status dictionary
        """
        return {
            "stt": self.stt.is_loaded() if self.stt else False,
            "llm": self.llm.is_loaded() if self.llm else False,
            "tts": self.tts.is_loaded() if self.tts else False,
            "pipeline": all([
                self.stt and self.stt.is_loaded(),
                self.llm and self.llm.is_loaded(),
                self.tts and self.tts.is_loaded()
            ])
        }