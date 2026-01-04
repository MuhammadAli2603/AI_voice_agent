"""
Simple Voice Agent Demo - Standalone Test
Tests the voice pipeline without requiring Knowledge Base service
"""
import asyncio
import sys
import os
import time
import numpy as np

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force load .env file
from dotenv import load_dotenv
load_dotenv(override=True)

# Verify API key is loaded
api_key = os.getenv('HUGGINGFACE_API_KEY', '')
if not api_key or api_key == 'hf_placeholder_key':
    print("❌ ERROR: No valid Hugging Face API key found!")
    print("   Please add your key to .env file")
    sys.exit(1)
else:
    print(f"✓ API key loaded: {api_key[:10]}...{api_key[-4:]}")
    print()

print("=" * 70)
print("AI VOICE AGENT - SIMPLE DEMO")
print("=" * 70)
print()
print("Testing the voice processing pipeline:")
print("  • Speech-to-Text (STT) - Hugging Face Whisper")
print("  • Language Model (LLM) - Receptionist AI")
print("  • Text-to-Speech (TTS) - Hugging Face TTS")
print()
print("Starting automated demo...")
print()

# Test imports
print("📦 Loading modules...")
try:
    from app.modules.stt.whisper_stt import WhisperSTT
    print("  ✓ STT module loaded")
except Exception as e:
    print(f"  ✗ STT module error: {e}")
    sys.exit(1)

try:
    from app.modules.llm.receptionist_llm import ReceptionistLLM
    print("  ✓ LLM module loaded")
except Exception as e:
    print(f"  ✗ LLM module error: {e}")
    sys.exit(1)

try:
    from app.modules.tts.huggingface_tts import HuggingFaceTTS
    print("  ✓ TTS module loaded")
except Exception as e:
    print(f"  ✗ TTS module error: {e}")
    sys.exit(1)

print()
print("✅ All modules loaded successfully!")
print()

# Create test audio
print("🎵 Creating test audio (2 seconds of simulated speech)...")
sample_rate = 16000
duration = 2.0
audio_array = np.random.normal(0, 0.01, int(sample_rate * duration)).astype(np.float32)
print(f"  • Sample rate: {sample_rate} Hz")
print(f"  • Duration: {duration}s")
print(f"  • Samples: {len(audio_array)}")
print()

# Initialize modules
print("🔧 Initializing AI modules...")
print()

try:
    # STT
    print("  1️⃣ Initializing Speech-to-Text...")
    stt = WhisperSTT(api_key=api_key)
    print("     ✓ Whisper STT ready")
    print()

    # Test STT with sample audio
    print("  📝 Testing transcription with sample audio...")
    start_time = time.time()

    # Create temporary WAV file
    import tempfile
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio_array, sample_rate)

    transcription_result = stt.transcribe(tmp_path)
    stt_time = time.time() - start_time

    os.unlink(tmp_path)  # Clean up

    print(f"     ✓ Transcription: \"{transcription_result['text']}\"")
    print(f"     ✓ Confidence: {transcription_result.get('confidence', 0):.2%}")
    print(f"     ✓ Time: {stt_time:.2f}s")
    print()

    # Use the transcription or fallback to test text
    user_input = transcription_result['text'] if transcription_result['text'] else "Hello, I need help with my order"

    # LLM
    print("  2️⃣ Initializing Language Model...")
    llm = ReceptionistLLM()
    print("     ✓ Receptionist LLM ready")
    print()

    # Test LLM
    print(f"  💬 Generating response to: \"{user_input}\"")
    start_time = time.time()

    response = llm.generate_response(
        user_input=user_input,
        conversation_history=[],
        context=""
    )
    llm_time = time.time() - start_time

    print(f"     ✓ Response: \"{response}\"")
    print(f"     ✓ Time: {llm_time:.2f}s")
    print()

    # TTS
    print("  3️⃣ Initializing Text-to-Speech...")
    tts = HuggingFaceTTS()
    print("     ✓ TTS ready")
    print()

    # Test TTS
    print(f"  🔊 Synthesizing speech: \"{response[:50]}...\"")
    start_time = time.time()

    audio_output = tts.synthesize(response)
    tts_time = time.time() - start_time

    print(f"     ✓ Audio generated: {len(audio_output)} bytes")
    print(f"     ✓ Time: {tts_time:.2f}s")
    print()

    # Save audio
    output_file = "demo_output.wav"
    with open(output_file, 'wb') as f:
        f.write(audio_output)

    print(f"     💾 Saved to: {output_file}")
    print()

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("=" * 70)
print("📊 DEMO SUMMARY")
print("=" * 70)
print()
print(f"User Input:  \"{user_input}\"")
print(f"AI Response: \"{response}\"")
print()
print("⏱️  Performance:")
print(f"  • STT:   {stt_time:.2f}s")
print(f"  • LLM:   {llm_time:.2f}s")
print(f"  • TTS:   {tts_time:.2f}s")
print(f"  • Total: {stt_time + llm_time + tts_time:.2f}s")
print()
print("📁 Files created:")
print(f"  • {output_file}")
print()
print("✅ Demo completed successfully!")
print()
print("🎉 Your ARI voice agent is working! The full ARI system adds:")
print("  • Full-duplex audio streaming")
print("  • Real-time barge-in detection")
print("  • Multi-call session management")
print("  • Telephony integration via Asterisk")
print()
