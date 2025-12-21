"""
Quick validation test for all modules (API-based)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.modules.stt.whisper_stt import WhisperSTT
from app.modules.llm.receptionist_llm import ReceptionistLLM
from app.modules.tts.huggingface_tts import HuggingFaceTTS


def test_stt_module():
    """Test STT module initialization"""
    print("\n🎤 Testing STT Module...")
    try:
        stt = WhisperSTT(model_name="openai/whisper-tiny", api_key="test")
        assert stt.is_loaded()
        print("   ✅ STT module OK")
        return True
    except Exception as e:
        print(f"   ❌ STT Error: {e}")
        return False


def test_llm_module():
    """Test LLM module initialization"""
    print("\n🤖 Testing LLM Module...")
    try:
        llm = ReceptionistLLM(
            model_name="microsoft/DialoGPT-small",
            api_key="test"
        )
        assert llm.is_loaded()
        print("   ✅ LLM module OK")
        return True
    except Exception as e:
        print(f"   ❌ LLM Error: {e}")
        return False


def test_tts_module():
    """Test TTS module initialization"""
    print("\n🔊 Testing TTS Module...")
    try:
        tts = HuggingFaceTTS(
            model_name="facebook/mms-tts-eng",
            api_key="test"
        )
        assert tts.is_loaded()
        print("   ✅ TTS module OK")
        return True
    except Exception as e:
        print(f"   ❌ TTS Error: {e}")
        return False


def main():
    """Run all module tests"""
    print("\n" + "="*60)
    print("🧪 AI Voice Agent - Module Validation Tests")
    print("="*60)
    
    results = {
        "STT": test_stt_module(),
        "LLM": test_llm_module(),
        "TTS": test_tts_module()
    }
    
    print("\n" + "="*60)
    print("📊 Test Results Summary")
    print("="*60)
    
    for module, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {module}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All modules initialized successfully!")
        print("\n📝 Note: These are structure tests only.")
        print("   For full API testing, set HUGGINGFACE_API_KEY in .env")
    else:
        print("⚠️  Some modules failed initialization.")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())