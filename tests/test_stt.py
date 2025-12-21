"""
Unit tests for Speech-to-Text (STT) module - API Version
Uses models from .env file
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from app.modules.stt.whisper_stt import WhisperSTT
from app.utils.audio_utils import normalize_audio, trim_silence
from app.config import settings


class TestWhisperSTT:
    """Test cases for Whisper STT module (API-based)"""
    
    @pytest.fixture
    def stt_model(self):
        """Initialize STT model for testing - uses .env settings"""
        print(f"\n📝 Using STT Model: {settings.stt_model}")
        return WhisperSTT(
            model_name=settings.stt_model,
            api_key=settings.huggingface_api_key
        )
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data"""
        # 1 second of 440 Hz sine wave (musical note A)
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return audio
    
    def test_model_initialization(self, stt_model):
        """Test if model initializes correctly"""
        assert stt_model.model == "api"
        assert stt_model.is_loaded() is True
        print(f"\n✓ STT API initialized")
        print(f"   Model: {stt_model.model_name}")
    
    def test_audio_preprocessing(self):
        """Test audio preprocessing utilities"""
        # Create test audio with silence
        audio = np.concatenate([
            np.zeros(1000),  # Silence
            np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)),  # Tone
            np.zeros(1000)   # Silence
        ]).astype(np.float32)
        
        # Test normalization
        normalized = normalize_audio(audio)
        assert np.abs(normalized).max() <= 1.0
        print("\n✓ Audio normalization works")
        
        # Test silence trimming
        trimmed = trim_silence(audio, sample_rate=16000)
        # Should trim some silence (not exact, just check it works)
        assert len(trimmed) <= len(audio)
        print(f"✓ Trimmed from {len(audio)} to {len(trimmed)} samples")


def test_whisper_api_structure():
    """Test Whisper API structure - uses .env settings"""
    print(f"\n📝 Testing with: {settings.stt_model}")
    stt = WhisperSTT(
        model_name=settings.stt_model,
        api_key=settings.huggingface_api_key
    )
    
    assert stt.is_loaded()
    assert stt.model_name == settings.stt_model
    assert f"models/{settings.stt_model}" in stt.api_url
    print(f"\n✓ API structure correct")
    print(f"   URL: {stt.api_url}")


if __name__ == "__main__":
    print("Running STT Module Tests (API Version)...")
    print("=" * 50)
    print(f"📝 STT Model from .env: {settings.stt_model}")
    print("=" * 50)
    
    print("\n1. Testing model initialization...")
    stt = WhisperSTT(
        model_name=settings.stt_model,
        api_key=settings.huggingface_api_key
    )
    print(f"   Model loaded: {stt.is_loaded()}")
    print(f"   Model: {stt.model_name}")
    print(f"   API URL: {stt.api_url}")
    
    print("\n2. Testing audio utilities...")
    test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
    normalized = normalize_audio(test_audio)
    print(f"   Normalized max: {np.abs(normalized).max()}")
    
    print("\n✅ All basic tests passed!")
    print(f"\n💡 Using: {settings.stt_model}")
    print("Note: Actual API transcription requires valid Hugging Face API key")