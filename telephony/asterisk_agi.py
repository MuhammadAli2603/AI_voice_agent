"""
Asterisk AGI (Asterisk Gateway Interface) integration for AI Voice Agent.
Handles incoming calls and bridges them to the AI voice processing pipeline.
"""
import os
import sys
import time
import base64
import asyncio
from asterisk.agi import AGI
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline.voice_pipeline import VoiceAgentPipeline
from telephony.call_session import CallSessionManager, CallState
from telephony.audio_bridge import AudioBridge
from telephony.barge_in import BargeInHandler, TTSPlaybackController
from app.utils.logger import log


class AsteriskAGIHandler:
    """
    Handles Asterisk AGI calls and routes them through AI voice pipeline.

    Flow:
    1. Answer call
    2. Detect company from DID/caller ID
    3. Create call session
    4. Conversation loop:
       - Record user speech (with barge-in support)
       - Transcribe via STT
       - Generate response via LLM (with KB context)
       - Synthesize via TTS
       - Play to caller (with interrupt detection)
    5. End call and cleanup
    """

    def __init__(self):
        """Initialize AGI handler"""
        self.agi = AGI()
        self.pipeline = VoiceAgentPipeline()
        self.session_manager = CallSessionManager()
        self.audio_bridge = AudioBridge(
            telephony_sample_rate=8000,
            telephony_codec='mulaw',
            ai_sample_rate=16000
        )

        # Temporary file directory
        self.temp_dir = "/tmp/voice_agent"
        os.makedirs(self.temp_dir, exist_ok=True)

        # Get call information from AGI environment
        self.call_id = self.agi.env.get('agi_uniqueid', 'unknown')
        self.caller_number = self.agi.env.get('agi_callerid', 'unknown')
        self.called_number = self.agi.env.get('agi_dnid', 'unknown')

        log.info(f"AGI Handler initialized for call {self.call_id}")

    async def handle_call(self):
        """
        Main call handling logic.
        Orchestrates the entire voice conversation.
        """
        session = None

        try:
            # Answer the call
            self.agi.answer()
            log.info(f"Call answered: {self.caller_number} → {self.called_number}")

            # Detect company context
            company_id = self._detect_company()
            log.info(f"Company detected: {company_id}")

            # Create call session
            session = await self.session_manager.create_session(
                call_id=self.call_id,
                caller_number=self.caller_number,
                called_number=self.called_number,
                company_id=company_id
            )

            # Welcome greeting
            await self._play_welcome(company_id)

            # Main conversation loop
            await self._conversation_loop(session)

            # Goodbye
            await self._play_goodbye()

            # End session
            await self.session_manager.end_session(session.session_id, reason="completed")

        except KeyboardInterrupt:
            log.info("Call interrupted by user")
            if session:
                await self.session_manager.end_session(session.session_id, reason="interrupted")

        except Exception as e:
            log.error(f"Error handling call: {e}")
            if session:
                session.add_error(str(e))
                await self.session_manager.end_session(session.session_id, reason="error")

            # Play error message to caller
            self._play_error_message()

        finally:
            # Cleanup
            self._cleanup()
            self.agi.hangup()

    async def _conversation_loop(self, session):
        """
        Main conversation loop with barge-in support.

        Args:
            session: Call session object
        """
        max_turns = 50  # Prevent infinite loops
        silence_timeout = 10000  # 10 seconds of silence
        no_input_count = 0
        max_no_input = 3

        for turn in range(max_turns):
            try:
                # Update session state
                session.state = CallState.IN_PROGRESS
                await self.session_manager.update_session(session)

                # Record user speech
                log.info(f"Turn {turn + 1}: Recording user speech...")
                audio_file = await self._record_user_speech(
                    filename=f"user_{self.call_id}_{turn}",
                    timeout=silence_timeout
                )

                # Check if recording succeeded
                if not audio_file or not os.path.exists(audio_file):
                    no_input_count += 1
                    if no_input_count >= max_no_input:
                        log.warning("No input from user, ending call")
                        self._play_prompt("no-input-goodbye")
                        break
                    continue

                # Reset no-input counter
                no_input_count = 0

                # Read audio file
                with open(audio_file, 'rb') as f:
                    audio_bytes = f.read()

                # Convert telephony audio to AI format
                audio_array, sr = self.audio_bridge.telephony_to_ai(audio_bytes)

                # Process through voice pipeline
                start_time = time.time()

                result = self.pipeline.process_audio(
                    audio_input=audio_array,
                    language='en',
                    conversation_history=self._build_conversation_history(session),
                    company_id=session.company_id
                )

                processing_time = time.time() - start_time

                # Extract results
                transcription = result['transcription']
                llm_response = result['llm_response']
                audio_base64 = result['audio_base64']
                timing = result['timing']

                log.info(f"User: {transcription}")
                log.info(f"Agent: {llm_response}")
                log.info(f"Processing: {processing_time:.2f}s")

                # Decode TTS audio
                tts_audio_bytes = base64.b64decode(audio_base64)

                # Convert AI audio to telephony format
                # First convert base64 -> numpy array
                import numpy as np
                import io
                import soundfile as sf
                tts_audio_array, _ = sf.read(io.BytesIO(tts_audio_bytes))
                telephony_audio = self.audio_bridge.ai_to_telephony(tts_audio_array)

                # Save response audio
                response_file = f"{self.temp_dir}/response_{self.call_id}_{turn}.raw"
                with open(response_file, 'wb') as f:
                    f.write(telephony_audio)

                # Play response to caller (with barge-in detection)
                interrupted = await self._play_response_with_barge_in(
                    audio_file=response_file,
                    session=session
                )

                # Add turn to session
                session.add_turn(
                    user_text=transcription,
                    agent_text=llm_response,
                    stt_latency=timing['stt'],
                    llm_latency=timing['llm'],
                    tts_latency=timing['tts'],
                    kb_chunks=result.get('kb_chunks', []),
                    confidence=result.get('kb_confidence', 0.0),
                    interrupted=interrupted
                )

                # Update session
                await self.session_manager.update_session(session)

                # Check for end conditions
                if self._should_end_call(transcription, llm_response):
                    log.info("End of conversation detected")
                    break

            except Exception as e:
                log.error(f"Error in conversation turn {turn}: {e}")
                session.add_error(str(e))
                # Try to continue
                continue

    async def _record_user_speech(
        self,
        filename: str,
        timeout: int = 10000,
        beep: bool = False
    ) -> Optional[str]:
        """
        Record user speech from call.

        Args:
            filename: Base filename (without extension)
            timeout: Max recording time in milliseconds
            beep: Play beep before recording

        Returns:
            Path to recorded file, or None if failed
        """
        filepath = f"{self.temp_dir}/{filename}"

        try:
            # Record file (AGI automatically adds .wav extension)
            # Format: RECORD FILE <filename> <format> <escape digits> <timeout> [OFFSET] [BEEP] [s=silence]
            result = self.agi.record_file(
                filepath,
                format='wav',
                escape_digits='#',  # User can press # to finish
                timeout=timeout,
                offset=0,
                beep=beep,
                s=2  # 2 seconds of silence ends recording
            )

            # Check recording status
            status = self.agi.get_variable('RECORD_STATUS')

            if status == 'HANGUP':
                log.warning("User hung up during recording")
                return None
            elif status == 'TIMEOUT':
                log.debug("Recording timeout (silence)")
                # This is normal - user finished speaking
            elif status == 'DTMF':
                log.debug("Recording ended by DTMF")

            # Verify file exists
            wav_file = f"{filepath}.wav"
            if os.path.exists(wav_file):
                file_size = os.path.getsize(wav_file)
                if file_size < 1000:  # Less than 1KB - probably empty
                    log.warning(f"Recording too small: {file_size} bytes")
                    return None

                return wav_file
            else:
                log.error(f"Recording file not found: {wav_file}")
                return None

        except Exception as e:
            log.error(f"Error recording user speech: {e}")
            return None

    async def _play_response_with_barge_in(
        self,
        audio_file: str,
        session
    ) -> bool:
        """
        Play AI response with barge-in detection.

        Args:
            audio_file: Path to audio file
            session: Call session

        Returns:
            True if interrupted, False otherwise
        """
        # TODO: Implement barge-in with streaming
        # For now, just play the file (no barge-in)

        try:
            # Play file (remove extension for Asterisk)
            audio_file_noext = audio_file.replace('.raw', '')
            self.agi.stream_file(audio_file_noext)

            return False  # Not interrupted

        except Exception as e:
            log.error(f"Error playing response: {e}")
            return False

    def _detect_company(self) -> str:
        """
        Detect company context from called number (DID).

        Returns:
            Company ID
        """
        # Map DIDs to companies
        did_map = {
            '18005551234': 'techstore',
            '18005555678': 'healthplus',
            '18005559999': 'edulearn'
        }

        # Clean DID (remove formatting)
        did = self.called_number.replace('-', '').replace(' ', '').replace('+', '')

        company_id = did_map.get(did, 'techstore')  # Default to techstore

        # Could also detect from caller ID or play IVR menu
        return company_id

    async def _play_welcome(self, company_id: str):
        """Play welcome greeting"""
        # TODO: Load company-specific greeting
        self.agi.stream_file('welcome')

    async def _play_goodbye(self):
        """Play goodbye message"""
        self.agi.stream_file('goodbye')

    def _play_error_message(self):
        """Play error message to caller"""
        try:
            self.agi.stream_file('error')
        except:
            pass

    def _play_prompt(self, prompt_name: str):
        """Play audio prompt"""
        try:
            self.agi.stream_file(prompt_name)
        except Exception as e:
            log.error(f"Error playing prompt {prompt_name}: {e}")

    def _build_conversation_history(self, session) -> list:
        """
        Build conversation history for LLM context.

        Args:
            session: Call session

        Returns:
            List of conversation turns
        """
        history = []
        for turn in session.conversation_history[-3:]:  # Last 3 turns
            history.append({
                'user': turn.user_text,
                'assistant': turn.agent_text
            })
        return history

    def _should_end_call(self, user_text: str, agent_text: str) -> bool:
        """
        Determine if call should end based on conversation.

        Args:
            user_text: User's last message
            agent_text: Agent's last response

        Returns:
            True if call should end
        """
        # Check for goodbye indicators
        goodbye_keywords = ['goodbye', 'bye', 'thank you', 'thanks', 'end call', 'hang up']

        user_lower = user_text.lower()
        agent_lower = agent_text.lower()

        for keyword in goodbye_keywords:
            if keyword in user_lower or keyword in agent_lower:
                return True

        return False

    def _cleanup(self):
        """Clean up temporary files"""
        try:
            # Remove temporary audio files for this call
            import glob
            for file in glob.glob(f"{self.temp_dir}/*_{self.call_id}_*"):
                try:
                    os.remove(file)
                except:
                    pass
        except Exception as e:
            log.error(f"Error during cleanup: {e}")


# Main entry point for Asterisk AGI
def main():
    """
    Main entry point called by Asterisk.
    Usage in extensions.conf:
      exten => 100,1,Answer()
      same => n,AGI(python,/path/to/asterisk_agi.py)
      same => n,Hangup()
    """
    try:
        handler = AsteriskAGIHandler()

        # Run async handler
        asyncio.run(handler.handle_call())

    except Exception as e:
        # Log to file since stdout goes to Asterisk
        with open('/var/log/asterisk/agi_error.log', 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Error: {e}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
