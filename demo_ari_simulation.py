"""
ARI Demo/Simulation Mode - Test the voice agent without Asterisk

This simulates incoming calls and tests the full pipeline:
- STT (Speech-to-Text)
- LLM (Language Model with KB)
- TTS (Text-to-Speech)
- Barge-in detection
- Session management

Usage:
    python demo_ari_simulation.py
"""
import asyncio
import sys
import os
import time
import base64
import numpy as np
import soundfile as sf
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.pipeline.voice_pipeline import VoiceAgentPipeline
from telephony.call_session import CallSessionManager, CallState
from app.utils.logger import log


class ARISimulator:
    """
    Simulates Asterisk ARI for testing without actual Asterisk installation.
    """

    def __init__(self):
        """Initialize simulator"""
        self.pipeline = VoiceAgentPipeline()
        self.session_manager = CallSessionManager()
        self.demo_audio_file = "demo_audio.wav"

        print("=" * 60)
        print("🎭 ARI DEMO MODE - Voice Agent Simulation")
        print("=" * 60)
        print()

    async def run_demo(self):
        """Run interactive demo"""

        # Create demo call session
        print("📞 Simulating incoming call...")
        print("   Caller: +1-555-0100")
        print("   Called: +1-800-555-1234 (TechStore)")
        print()

        session = await self.session_manager.create_session(
            call_id="DEMO-CALL-001",
            caller_number="+1-555-0100",
            called_number="+1-800-555-1234",
            company_id="techstore"
        )

        print(f"✅ Call session created: {session.session_id}")
        print()

        # Welcome message
        print("🤖 AI Agent: Welcome to TechStore! How can I help you today?")
        print()

        # Conversation loop
        turn = 0
        while turn < 5:  # Max 5 turns for demo
            turn += 1
            print(f"--- Turn {turn} ---")
            print()

            # Get user input
            print("🎤 You: ", end="", flush=True)
            user_text = input()

            if not user_text or user_text.lower() in ['quit', 'exit', 'goodbye', 'bye']:
                print()
                print("🤖 AI Agent: Thank you for contacting TechStore! Have a great day!")
                break

            print()
            print("⏳ Processing your request...")
            print("   1️⃣ Converting speech to text (simulated)...")
            await asyncio.sleep(0.3)

            print("   2️⃣ Querying knowledge base...")
            await asyncio.sleep(0.5)

            print("   3️⃣ Generating AI response...")

            # Create simulated audio input
            audio_array = self._create_simulated_audio(user_text)

            # Process through pipeline
            start_time = time.time()

            try:
                result = self.pipeline.process_audio(
                    audio_input=audio_array,
                    language='en',
                    conversation_history=self._build_conversation_history(session),
                    company_id=session.company_id
                )

                processing_time = time.time() - start_time

                # Display results
                print()
                print("📊 Processing Results:")
                print(f"   • Transcription: {result['transcription']}")
                print(f"   • KB Confidence: {result.get('kb_confidence', 0):.2%}")
                print(f"   • Processing Time: {processing_time:.2f}s")
                print(f"   • STT: {result['timing']['stt']:.2f}s")
                print(f"   • LLM: {result['timing']['llm']:.2f}s")
                print(f"   • TTS: {result['timing']['tts']:.2f}s")
                print()

                if result.get('kb_chunks'):
                    print("📚 Knowledge Base Chunks Used:")
                    for i, chunk in enumerate(result['kb_chunks'][:2], 1):
                        preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                        print(f"   {i}. {preview}")
                    print()

                print(f"🤖 AI Agent: {result['llm_response']}")
                print()

                # Add turn to session
                session.add_turn(
                    user_text=result['transcription'],
                    agent_text=result['llm_response'],
                    stt_latency=result['timing']['stt'],
                    llm_latency=result['timing']['llm'],
                    tts_latency=result['timing']['tts'],
                    kb_chunks=result.get('kb_chunks', []),
                    confidence=result.get('kb_confidence', 0.0),
                    interrupted=False
                )

                # Save audio response
                if result.get('audio_base64'):
                    self._save_audio_response(result['audio_base64'], turn)
                    print(f"💾 Audio saved: demo_response_{turn}.wav")
                    print()

            except Exception as e:
                print(f"❌ Error processing: {e}")
                print()
                session.add_error(str(e))

        # End session
        print()
        print("=" * 60)
        print("📊 Call Summary")
        print("=" * 60)
        print(f"Duration: {session.duration_seconds():.1f} seconds")
        print(f"Turns: {session.current_turn}")
        print(f"Avg STT Latency: {session.avg_stt_latency:.0f}ms")
        print(f"Avg LLM Latency: {session.avg_llm_latency:.0f}ms")
        print(f"Avg TTS Latency: {session.avg_tts_latency:.0f}ms")
        print(f"Interruptions: {session.total_interrupts}")
        print(f"Errors: {session.total_errors}")
        print()

        await self.session_manager.end_session(session.session_id, reason="completed")

        print("✅ Demo complete!")
        print()
        print("📁 Files created:")
        print("   • Session log: logs/calls_*.jsonl")
        for i in range(1, turn + 1):
            print(f"   • Audio response {i}: demo_response_{i}.wav")
        print()

    def _create_simulated_audio(self, text: str) -> np.ndarray:
        """
        Create simulated audio input for testing.

        Args:
            text: Text to simulate as audio

        Returns:
            Audio array (silent, for demo purposes)
        """
        # Create 2 seconds of silent audio at 16kHz
        sample_rate = 16000
        duration = 2.0
        samples = int(sample_rate * duration)

        # Create silence with slight noise
        audio = np.random.normal(0, 0.001, samples).astype(np.float32)

        return audio

    def _save_audio_response(self, audio_base64: str, turn: int):
        """Save TTS audio response to file"""
        try:
            audio_bytes = base64.b64decode(audio_base64)
            filename = f"demo_response_{turn}.wav"

            with open(filename, 'wb') as f:
                f.write(audio_bytes)

        except Exception as e:
            log.error(f"Error saving audio: {e}")

    def _build_conversation_history(self, session) -> list:
        """Build conversation history for LLM context"""
        history = []
        for turn in session.conversation_history[-3:]:
            history.append({
                'user': turn.user_text,
                'assistant': turn.agent_text
            })
        return history


async def main():
    """Main entry point for demo"""

    print()
    print("This demo simulates the ARI voice agent without requiring Asterisk.")
    print("You can type your questions as if you were speaking on the phone.")
    print()
    print("The system will:")
    print("  1. Process your text through the AI pipeline")
    print("  2. Query the knowledge base")
    print("  3. Generate an AI response")
    print("  4. Create audio files (TTS)")
    print("  5. Track metrics and session data")
    print()
    print("Type 'quit', 'exit', or 'goodbye' to end the call.")
    print()
    input("Press Enter to start the demo call...")
    print()

    # Run simulator
    simulator = ARISimulator()
    await simulator.run_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
