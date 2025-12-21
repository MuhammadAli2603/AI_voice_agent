"""
Unit tests for Language Model (LLM) module - API Version
Uses models from .env file
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from app.modules.llm.receptionist_llm import ReceptionistLLM
from app.config import settings


class TestReceptionistLLM:
    """Test cases for Receptionist LLM module (API-based)"""
    
    @pytest.fixture
    def llm_model(self):
        """Initialize LLM model for testing - uses .env settings"""
        print(f"\n🤖 Using LLM Model: {settings.llm_model}")
        return ReceptionistLLM(
            model_name=settings.llm_model,
            api_key=settings.huggingface_api_key,
            max_length=settings.llm_max_length,
            temperature=settings.llm_temperature,
            top_p=settings.llm_top_p
        )
    
    def test_model_initialization(self, llm_model):
        """Test if model initializes correctly"""
        assert llm_model.model == "api"
        assert llm_model.tokenizer == "api"
        assert llm_model.is_loaded() is True
        print(f"\n✓ LLM API initialized")
        print(f"   Model: {llm_model.model_name}")
        print(f"   Temperature: {llm_model.temperature}")
    
    def test_conversation_structure(self, llm_model):
        """Test conversation history structure"""
        assert llm_model.conversation_history == []
        print("\n✓ Conversation history initialized")
    
    def test_prompt_building(self, llm_model):
        """Test prompt building"""
        prompt = llm_model._build_prompt("Hello!")
        assert "Hello!" in prompt
        assert "receptionist" in prompt.lower() or "assistant" in prompt.lower()
        print("\n✓ Prompt building works")
    
    def test_response_cleaning(self, llm_model):
        """Test response cleaning"""
        # Test with prefix
        response = llm_model._clean_response("Assistant: Hello there!")
        assert response == "Hello there!"
        
        # Test without ending punctuation
        response = llm_model._clean_response("Hello")
        assert response == "Hello."
        
        print("\n✓ Response cleaning works")
    
    def test_reset_conversation(self, llm_model):
        """Test conversation reset"""
        llm_model.conversation_history = [{"user": "test", "assistant": "test"}]
        llm_model.reset_conversation()
        assert llm_model.conversation_history == []
        print("\n✓ Conversation reset works")


def test_llm_api_structure():
    """Test LLM API structure - uses .env settings"""
    print(f"\n🤖 Testing with: {settings.llm_model}")
    llm = ReceptionistLLM(
        model_name=settings.llm_model,
        api_key=settings.huggingface_api_key,
        max_length=settings.llm_max_length,
        temperature=settings.llm_temperature
    )
    
    assert llm.is_loaded()
    assert llm.model_name == settings.llm_model
    assert f"models/{settings.llm_model}" in llm.api_url
    print(f"\n✓ LLM API structure correct")
    print(f"   URL: {llm.api_url}")


if __name__ == "__main__":
    print("Running LLM Module Tests (API Version)...")
    print("=" * 50)
    print(f"🤖 LLM Model from .env: {settings.llm_model}")
    print("=" * 50)
    
    print("\n1. Testing model initialization...")
    llm = ReceptionistLLM(
        model_name=settings.llm_model,
        api_key=settings.huggingface_api_key
    )
    print(f"   Model loaded: {llm.is_loaded()}")
    print(f"   Model: {llm.model_name}")
    print(f"   API URL: {llm.api_url}")
    
    print("\n2. Testing prompt building...")
    prompt = llm._build_prompt("Hello!")
    print(f"   Prompt length: {len(prompt)} chars")
    
    print("\n3. Testing response cleaning...")
    cleaned = llm._clean_response("Assistant: Hi there")
    print(f"   Cleaned: '{cleaned}'")
    
    print("\n✅ All basic tests passed!")
    print(f"\n💡 Using: {settings.llm_model}")
    print("Note: Actual response generation requires valid Hugging Face API key")