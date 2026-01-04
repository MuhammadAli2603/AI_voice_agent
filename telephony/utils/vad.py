"""
Voice Activity Detection (VAD) for barge-in support.
Detects when caller is speaking during AI response playback.
"""
import numpy as np
import torch
from typing import Optional, Tuple
from enum import Enum
import time


class VADMode(Enum):
    """VAD sensitivity modes"""
    QUALITY = 0      # High quality (less aggressive)
    LOW_BITRATE = 1  # Optimized for low bitrate
    AGGRESSIVE = 2   # Most aggressive (best for noisy environments)
    VERY_AGGRESSIVE = 3  # Maximum sensitivity


class VoiceActivityDetector:
    """
    Voice Activity Detection using Silero VAD.

    Detects speech in audio stream to enable caller interruption (barge-in).
    Uses pre-trained Silero VAD model for high accuracy in noisy phone environments.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        padding_duration_ms: int = 30
    ):
        """
        Initialize VAD with Silero model.

        Args:
            sample_rate: Audio sample rate (8000 or 16000)
            threshold: Speech detection threshold (0.0-1.0, default 0.5)
            min_speech_duration_ms: Minimum speech duration to trigger
            min_silence_duration_ms: Minimum silence to confirm end of speech
            padding_duration_ms: Padding around speech segments
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.padding_duration_ms = padding_duration_ms

        # Load Silero VAD model
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )

        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils

        # Initialize iterator for streaming detection
        self.vad_iterator = self.VADIterator(
            self.model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=padding_duration_ms
        )

        # State tracking
        self.is_speech_active = False
        self.speech_start_time = None
        self.last_speech_time = None

        print(f"✓ Silero VAD initialized (threshold={threshold}, sample_rate={sample_rate}Hz)")

    def detect_speech(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """
        Detect speech in audio chunk.

        Args:
            audio_chunk: Audio data as numpy array (float32, -1 to 1)

        Returns:
            Tuple of (is_speech: bool, confidence: float)
        """
        # Convert to torch tensor
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk).float()
        else:
            audio_tensor = audio_chunk

        # Get VAD prediction
        speech_dict = self.vad_iterator(audio_tensor, return_seconds=False)

        if speech_dict:
            # Speech detected
            confidence = speech_dict.get('confidence', 0.0)
            return True, confidence
        else:
            # No speech
            return False, 0.0

    def is_speech_in_chunk(self, audio_chunk: bytes, format: str = 'int16') -> bool:
        """
        Simple binary speech detection for barge-in.

        Args:
            audio_chunk: Raw audio bytes
            format: Audio format ('int16' or 'float32')

        Returns:
            True if speech detected, False otherwise
        """
        # Convert bytes to numpy array
        if format == 'int16':
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            # Normalize to float32 (-1 to 1)
            audio_array = audio_array.astype(np.float32) / 32768.0
        else:
            audio_array = np.frombuffer(audio_chunk, dtype=np.float32)

        # Detect speech
        is_speech, confidence = self.detect_speech(audio_array)

        # Update state
        if is_speech:
            if not self.is_speech_active:
                self.speech_start_time = time.time()
                self.is_speech_active = True
            self.last_speech_time = time.time()
        else:
            # Check if silence long enough to end speech
            if self.is_speech_active and self.last_speech_time:
                silence_duration = (time.time() - self.last_speech_time) * 1000
                if silence_duration > self.min_silence_duration_ms:
                    self.is_speech_active = False

        return is_speech

    def get_speech_segments(self, audio: np.ndarray) -> list:
        """
        Get timestamps of speech segments in audio.

        Args:
            audio: Full audio array

        Returns:
            List of speech segment dicts with 'start' and 'end' timestamps
        """
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()

        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms
        )

        return speech_timestamps

    def reset(self):
        """Reset VAD state for new conversation"""
        self.vad_iterator.reset_states()
        self.is_speech_active = False
        self.speech_start_time = None
        self.last_speech_time = None


class WebRTCVAD:
    """
    Lightweight VAD using WebRTC (alternative to Silero).
    Faster but less accurate in noisy environments.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        mode: VADMode = VADMode.AGGRESSIVE
    ):
        """
        Initialize WebRTC VAD.

        Args:
            sample_rate: Must be 8000, 16000, 32000, or 48000
            mode: VAD aggressiveness mode
        """
        import webrtcvad

        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Sample rate must be 8000, 16000, 32000, or 48000, got {sample_rate}")

        self.vad = webrtcvad.Vad(mode.value)
        self.sample_rate = sample_rate
        self.frame_duration_ms = 30  # WebRTC supports 10, 20, or 30 ms
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)

        print(f"✓ WebRTC VAD initialized (mode={mode.name}, sample_rate={sample_rate}Hz)")

    def is_speech(self, audio_frame: bytes) -> bool:
        """
        Detect speech in audio frame.

        Args:
            audio_frame: Raw audio bytes (must be 16-bit PCM)

        Returns:
            True if speech detected, False otherwise
        """
        # Frame must be exact size (10, 20, or 30ms of audio)
        expected_size = self.frame_size * 2  # 2 bytes per sample (16-bit)

        if len(audio_frame) != expected_size:
            # Pad or truncate
            if len(audio_frame) < expected_size:
                audio_frame = audio_frame + b'\x00' * (expected_size - len(audio_frame))
            else:
                audio_frame = audio_frame[:expected_size]

        return self.vad.is_speech(audio_frame, self.sample_rate)


# Example usage and testing
if __name__ == "__main__":
    import soundfile as sf

    # Test Silero VAD
    print("Testing Silero VAD...")
    vad = VoiceActivityDetector(sample_rate=16000, threshold=0.5)

    # Load test audio (replace with actual file)
    # audio, sr = sf.read('test_audio.wav')
    # segments = vad.get_speech_segments(audio)
    # print(f"Found {len(segments)} speech segments")

    # Test streaming detection
    # for i in range(0, len(audio), 1600):  # 100ms chunks
    #     chunk = audio[i:i+1600]
    #     is_speech = vad.is_speech_in_chunk(chunk.tobytes(), format='float32')
    #     if is_speech:
    #         print(f"Speech detected at {i/sr:.2f}s")

    print("✓ VAD module ready")
