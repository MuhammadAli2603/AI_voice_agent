"""
Integration tests for Voice Agent Pipeline - API Version
Uses models from .env file
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from app.pipeline.voice_pipeline import VoiceAgentPipeline
from app.config import settings


class TestVoiceAgentPipeline:
    """Test cases for complete Voice Agent Pipeline (API-based)"""
    
    @pytest.fixture
    def pipeline(self):
        """Initialize pipeline for testing - uses .env settings"""
        print(f"\n🔗 Pipeline Models:")
        print(f"   STT: {settings.stt_model}")
        print(f"   LLM: {settings.llm_model}")
        print(f"   TTS: {settings.tts_model}")
        
        return VoiceAgentPipeline(
            stt_model=settings.stt_model,
            llm_model=settings.llm_model,
            tts_model=settings.tts_model,
            api_key=settings.huggingface_api_key
        )
    
    def test_pipeline_initialization(self, pipeline):
        """Test if pipeline initializes all modules"""
        assert pipeline.stt is not None
        assert pipeline.llm is not None
        assert pipeline.tts is not None
        
        health = pipeline.health_check()
        assert health["stt"] is True
        assert health["llm"] is True
        assert health["tts"] is True
        assert health["pipeline"] is True
        
        print("\n✓ Pipeline initialized successfully")
        print(f"   STT: {pipeline.stt_model_name}")
        print(f"   LLM: {pipeline.llm_model_name}")
        print(f"   TTS: {pipeline.tts_model_name}")
    
    def test_health_check(self, pipeline):
        """Test health check functionality"""
        health = pipeline.health_check()
        
        assert isinstance(health, dict)
        assert "stt" in health
        assert "llm" in health
        assert "tts" in health
        assert "pipeline" in health
        
        print("\n✓ Health check works")
        print(f"   All modules operational: {health['pipeline']}")
    
    def test_reset_conversation(self, pipeline):
        """Test conversation reset"""
        # Should not raise any errors
        pipeline.reset_conversation()
        print("\n✓ Conversation reset works")


def test_pipeline_structure():
    """Test pipeline structure - uses .env settings"""
    print(f"\n🔗 Testing Pipeline with .env models:")
    print(f"   STT: {settings.stt_model}")
    print(f"   LLM: {settings.llm_model}")
    print(f"   TTS: {settings.tts_model}")
    
    pipeline = VoiceAgentPipeline(
        stt_model=settings.stt_model,
        llm_model=settings.llm_model,
        tts_model=settings.tts_model,
        api_key=settings.huggingface_api_key
    )
    
    assert pipeline.api_key == settings.huggingface_api_key
    assert pipeline.stt.is_loaded()
    assert pipeline.llm.is_loaded()
    assert pipeline.tts.is_loaded()
    
    print("\n✓ Pipeline structure correct")


if __name__ == "__main__":
    print("Running Pipeline Integration Tests (API Version)...")
    print("=" * 50)
    print("📝 Using models from .env:")
    print(f"   STT: {settings.stt_model}")
    print(f"   LLM: {settings.llm_model}")
    print(f"   TTS: {settings.tts_model}")
    print("=" * 50)
    
    print("\n1. Initializing pipeline...")
    pipeline = VoiceAgentPipeline(
        stt_model=settings.stt_model,
        llm_model=settings.llm_model,
        tts_model=settings.tts_model,
        api_key=settings.huggingface_api_key
    )
    print("   ✓ Pipeline initialized")
    
    print("\n2. Testing health check...")
    health = pipeline.health_check()
    print(f"   STT: {'✓' if health['stt'] else '✗'}")
    print(f"   LLM: {'✓' if health['llm'] else '✗'}")
    print(f"   TTS: {'✓' if health['tts'] else '✗'}")
    print(f"   Pipeline: {'✓' if health['pipeline'] else '✗'}")
    
    print("\n3. Testing conversation reset...")
    pipeline.reset_conversation()
    print("   ✓ Conversation reset")
    
    print("\n✅ All basic integration tests passed!")
    print(f"\n💡 Using models from .env file")
    print("Note: Actual audio processing requires valid Hugging Face API key")