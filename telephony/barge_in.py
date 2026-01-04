"""
Barge-in (interruption) handler for voice agent.
Detects when caller interrupts AI response and handles gracefully.
"""
import asyncio
import numpy as np
from typing import Optional, Callable
from enum import Enum
from telephony.utils.vad import VoiceActivityDetector, WebRTCVAD
from app.utils.logger import log


class InterruptState(Enum):
    """Interrupt detection states"""
    IDLE = "idle"                   # Not monitoring
    MONITORING = "monitoring"        # Listening for interrupts
    INTERRUPTED = "interrupted"      # Interrupt detected
    PROCESSING = "processing"        # Processing interrupt


class BargeInHandler:
    """
    Handles caller interruptions during AI speech.

    When the AI is speaking:
    1. Monitors incoming audio stream
    2. Detects caller speech (VAD)
    3. Cancels TTS playback immediately
    4. Captures interrupting speech
    5. Processes as new user input
    """

    def __init__(
        self,
        vad_threshold: float = 0.5,
        min_interrupt_duration_ms: int = 300,
        use_advanced_vad: bool = True
    ):
        """
        Initialize barge-in handler.

        Args:
            vad_threshold: Speech detection threshold (0.0-1.0)
            min_interrupt_duration_ms: Minimum speech duration to trigger interrupt
            use_advanced_vad: Use Silero VAD (True) or WebRTC VAD (False)
        """
        self.vad_threshold = vad_threshold
        self.min_interrupt_duration_ms = min_interrupt_duration_ms

        # Initialize VAD
        if use_advanced_vad:
            self.vad = VoiceActivityDetector(
                threshold=vad_threshold,
                min_speech_duration_ms=min_interrupt_duration_ms
            )
        else:
            self.vad = WebRTCVAD()

        # State
        self.state = InterruptState.IDLE
        self.interrupt_buffer = []  # Buffer for interrupt audio
        self.monitoring_task: Optional[asyncio.Task] = None
        self.tts_cancel_callback: Optional[Callable] = None

        # Metrics
        self.total_interrupts = 0
        self.false_positives = 0

        log.info(f"BargeInHandler initialized (threshold={vad_threshold})")

    async def start_monitoring(
        self,
        audio_stream,
        on_interrupt: Callable,
        on_tts_cancel: Callable
    ):
        """
        Start monitoring for interruptions.

        Args:
            audio_stream: AsyncIterator yielding audio chunks
            on_interrupt: Callback when interrupt detected (async)
            on_tts_cancel: Callback to cancel TTS playback (async)
        """
        self.state = InterruptState.MONITORING
        self.tts_cancel_callback = on_tts_cancel
        self.interrupt_buffer = []

        log.info("Started monitoring for caller interruptions")

        # Start monitoring task
        self.monitoring_task = asyncio.create_task(
            self._monitor_loop(audio_stream, on_interrupt)
        )

    async def stop_monitoring(self):
        """Stop monitoring for interruptions"""
        self.state = InterruptState.IDLE

        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        log.info("Stopped monitoring for interruptions")

    async def _monitor_loop(self, audio_stream, on_interrupt: Callable):
        """
        Main monitoring loop.
        Checks each audio chunk for speech during AI playback.
        """
        speech_start_time = None
        consecutive_speech_chunks = 0
        required_chunks = int(self.min_interrupt_duration_ms / 20)  # Assuming 20ms chunks

        try:
            async for audio_chunk in audio_stream:
                if self.state != InterruptState.MONITORING:
                    break

                # Detect speech in chunk
                is_speech = self._detect_speech(audio_chunk)

                if is_speech:
                    consecutive_speech_chunks += 1

                    # Buffer the audio
                    self.interrupt_buffer.append(audio_chunk)

                    # Check if sustained speech (interrupt threshold reached)
                    if consecutive_speech_chunks >= required_chunks:
                        # INTERRUPT DETECTED!
                        await self._handle_interrupt(on_interrupt)
                        break
                else:
                    # Reset on silence
                    if consecutive_speech_chunks > 0 and consecutive_speech_chunks < required_chunks:
                        # False alarm - speech was too short
                        self.false_positives += 1

                    consecutive_speech_chunks = 0
                    self.interrupt_buffer = []

        except asyncio.CancelledError:
            log.debug("Monitoring loop cancelled")
        except Exception as e:
            log.error(f"Error in monitoring loop: {e}")

    def _detect_speech(self, audio_chunk: bytes) -> bool:
        """
        Detect speech in audio chunk.

        Args:
            audio_chunk: Raw audio bytes

        Returns:
            True if speech detected
        """
        try:
            if isinstance(self.vad, VoiceActivityDetector):
                # Silero VAD
                return self.vad.is_speech_in_chunk(audio_chunk, format='int16')
            else:
                # WebRTC VAD
                return self.vad.is_speech(audio_chunk)
        except Exception as e:
            log.error(f"VAD error: {e}")
            return False

    async def _handle_interrupt(self, on_interrupt: Callable):
        """
        Handle detected interrupt.

        Args:
            on_interrupt: Callback function to invoke
        """
        self.state = InterruptState.INTERRUPTED
        self.total_interrupts += 1

        log.info(f"🛑 INTERRUPT DETECTED! (total: {self.total_interrupts})")

        # 1. Cancel TTS playback immediately
        if self.tts_cancel_callback:
            try:
                await self.tts_cancel_callback()
                log.info("✓ TTS playback cancelled")
            except Exception as e:
                log.error(f"Error cancelling TTS: {e}")

        # 2. Prepare interrupt audio for processing
        interrupt_audio = self._concatenate_buffer()

        # 3. Invoke callback with interrupt audio
        self.state = InterruptState.PROCESSING
        try:
            await on_interrupt(interrupt_audio)
        except Exception as e:
            log.error(f"Error processing interrupt: {e}")
        finally:
            self.state = InterruptState.IDLE
            self.interrupt_buffer = []

    def _concatenate_buffer(self) -> bytes:
        """
        Concatenate buffered audio chunks.

        Returns:
            Combined audio as bytes
        """
        if not self.interrupt_buffer:
            return b''

        return b''.join(self.interrupt_buffer)

    def get_stats(self) -> dict:
        """
        Get interrupt statistics.

        Returns:
            Dict with metrics
        """
        return {
            'total_interrupts': self.total_interrupts,
            'false_positives': self.false_positives,
            'state': self.state.value,
            'buffer_size': len(self.interrupt_buffer)
        }

    def reset_stats(self):
        """Reset interrupt counters"""
        self.total_interrupts = 0
        self.false_positives = 0


class TTSPlaybackController:
    """
    Controls TTS playback with interrupt capability.
    Allows immediate cancellation when caller speaks.
    """

    def __init__(self):
        """Initialize playback controller"""
        self.is_playing = False
        self.current_audio: Optional[bytes] = None
        self.playback_task: Optional[asyncio.Task] = None
        self.cancel_event = asyncio.Event()

    async def play_audio(
        self,
        audio_data: bytes,
        play_chunk_callback: Callable,
        chunk_size: int = 160  # 20ms @ 8kHz
    ):
        """
        Play audio with cancellation support.

        Args:
            audio_data: Audio to play
            play_chunk_callback: Async function to send audio chunk to telephony
            chunk_size: Bytes per chunk
        """
        self.is_playing = True
        self.current_audio = audio_data
        self.cancel_event.clear()

        try:
            # Stream audio in chunks
            for i in range(0, len(audio_data), chunk_size):
                # Check if cancelled
                if self.cancel_event.is_set():
                    log.info("TTS playback cancelled")
                    break

                chunk = audio_data[i:i + chunk_size]
                await play_chunk_callback(chunk)

                # Small delay for real-time playback
                await asyncio.sleep(0.02)  # 20ms

            log.debug("TTS playback completed")

        except asyncio.CancelledError:
            log.info("TTS playback task cancelled")
        except Exception as e:
            log.error(f"Error during TTS playback: {e}")
        finally:
            self.is_playing = False
            self.current_audio = None

    async def cancel_playback(self):
        """
        Cancel current playback immediately.
        Called when interrupt detected.
        """
        if self.is_playing:
            log.info("Cancelling TTS playback...")
            self.cancel_event.set()

            if self.playback_task and not self.playback_task.done():
                self.playback_task.cancel()
                try:
                    await self.playback_task
                except asyncio.CancelledError:
                    pass

            self.is_playing = False

    def is_currently_playing(self) -> bool:
        """Check if audio is currently playing"""
        return self.is_playing


# Example usage
if __name__ == "__main__":
    async def test_barge_in():
        """Test barge-in detection"""

        # Callback when interrupt detected
        async def handle_interrupt(audio_data: bytes):
            print(f"Interrupt detected! Audio size: {len(audio_data)} bytes")
            # Process the interrupting speech...

        # Callback to cancel TTS
        async def cancel_tts():
            print("Cancelling TTS playback...")

        # Initialize handler
        handler = BargeInHandler(vad_threshold=0.5)

        # Simulate audio stream
        async def fake_audio_stream():
            """Generate fake audio chunks"""
            for i in range(100):
                # Simulate 20ms chunks of audio
                yield b'\x00' * 160
                await asyncio.sleep(0.02)

        # Start monitoring
        await handler.start_monitoring(
            audio_stream=fake_audio_stream(),
            on_interrupt=handle_interrupt,
            on_tts_cancel=cancel_tts
        )

        # Let it run for a bit
        await asyncio.sleep(2)

        # Stop monitoring
        await handler.stop_monitoring()

        print(f"Stats: {handler.get_stats()}")

    # Run test
    # asyncio.run(test_barge_in())
    print("✓ Barge-in handler ready")
