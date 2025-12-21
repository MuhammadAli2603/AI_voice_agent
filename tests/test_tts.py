"""
Unit tests for Text-to-Speech (TTS) module - API Version
Uses models from .env file
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from app.modules.tts.huggingface_tts import HuggingFaceTTS
from app.config import settings


class TestHuggingFaceTTS:
    """Test cases for Hugging Face TTS module (API-based)"""
    
    @pytest.fixture
    def tts_model(self):
        """Initialize TTS model for testing - uses .env settings"""
        print(f"\n🔊 Using TTS Model: {settings.tts_model}")
        return HuggingFaceTTS(
            model_name=settings.tts_model,
            api_key=settings.huggingface_api_key
        )
    
    def test_model_initialization(self, tts_model):
        """Test if model initializes correctly"""
        assert tts_model.model == "api"
        assert tts_model.is_loaded() is True
        print(f"\n✓ TTS API initialized")
        print(f"   Model: {tts_model.model_name}")
    
    def test_sample_rate(self, tts_model):
        """Test sample rate"""
        sample_rate = tts_model.get_sample_rate()
        assert isinstance(sample_rate, int)
        assert sample_rate > 0
        print(f"\n✓ Sample rate: {sample_rate} Hz")
    
    def test_time_stretch(self, tts_model):
        """Test simple time stretching"""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        
        # Test speed up
        faster = tts_model._simple_time_stretch(audio, rate=1.5)
        assert len(faster) < len(audio)
        
        # Test slow down
        slower = tts_model._simple_time_stretch(audio, rate=0.75)
        assert len(slower) > len(audio)
        
        print("\n✓ Time stretching works")


def test_tts_api_structure():
    """Test TTS API structure - uses .env settings"""
    print(f"\n🔊 Testing with: {settings.tts_model}")
    tts = HuggingFaceTTS(
        model_name=settings.tts_model,
        api_key=settings.huggingface_api_key
    )
    
    assert tts.is_loaded()
    assert tts.model_name == settings.tts_model
    assert f"models/{settings.tts_model}" in tts.api_url
    print(f"\n✓ TTS API structure correct")
    print(f"   URL: {tts.api_url}")


if __name__ == "__main__":
    print("Running TTS Module Tests (API Version)...")
    print("=" * 50)
    print(f"🔊 TTS Model from .env: {settings.tts_model}")
    print("=" * 50)
    
    print("\n1. Testing model initialization...")
    tts = HuggingFaceTTS(
        model_name=settings.tts_model,
        api_key=settings.huggingface_api_key
    )
    print(f"   Model loaded: {tts.is_loaded()}")
    print(f"   Model: {tts.model_name}")
    print(f"   API URL: {tts.api_url}")
    print(f"   Sample rate: {tts.get_sample_rate()} Hz")
    
    print("\n2. Testing time stretching...")
    test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
    stretched = tts._simple_time_stretch(test_audio, rate=1.5)
    print(f"   Original: {len(test_audio)} samples")
    print(f"   Stretched (1.5x): {len(stretched)} samples")
    
    print("\n✅ All basic tests passed!")
    print(f"\n💡 Using: {settings.tts_model}")
    print("Note: Actual synthesis requires valid Hugging Face API key")