"""
Audio format conversion bridge for telephony integration.
Handles conversion between telephony formats (μ-law, A-law) and AI model formats (16kHz PCM).
"""
import io
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from typing import Tuple, Union
from app.utils.logger import log


class AudioBridge:
    """
    Bidirectional audio format conversion for telephony.

    Telephony Systems:
    - Sample rate: 8kHz (narrow-band) or 16kHz (wide-band)
    - Codec: μ-law (G.711u) or A-law (G.711a)
    - Channels: Mono

    AI Models (Whisper, TTS):
    - Sample rate: 16kHz
    - Format: Linear PCM (16-bit)
    - Channels: Mono
    """

    def __init__(
        self,
        telephony_sample_rate: int = 8000,
        telephony_codec: str = 'mulaw',  # 'mulaw' or 'alaw'
        ai_sample_rate: int = 16000
    ):
        """
        Initialize audio bridge.

        Args:
            telephony_sample_rate: Telephony system sample rate (8000 or 16000)
            telephony_codec: Telephony codec ('mulaw', 'alaw', or 'linear16')
            ai_sample_rate: AI model sample rate (typically 16000)
        """
        self.telephony_sample_rate = telephony_sample_rate
        self.telephony_codec = telephony_codec
        self.ai_sample_rate = ai_sample_rate

        log.info(
            f"AudioBridge initialized: "
            f"{telephony_codec}@{telephony_sample_rate}Hz ↔ PCM@{ai_sample_rate}Hz"
        )

    def telephony_to_ai(
        self,
        audio_data: bytes,
        format: str = None
    ) -> Tuple[np.ndarray, int]:
        """
        Convert telephony audio to AI model format.

        Args:
            audio_data: Raw audio bytes from telephony system
            format: Audio format override (if different from init)

        Returns:
            Tuple of (audio_array: np.ndarray, sample_rate: int)
        """
        codec = format or self.telephony_codec

        try:
            # Convert based on codec
            if codec == 'mulaw':
                audio_array = self._mulaw_to_linear(audio_data)
            elif codec == 'alaw':
                audio_array = self._alaw_to_linear(audio_data)
            elif codec == 'linear16':
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            else:
                raise ValueError(f"Unsupported codec: {codec}")

            # Convert to float32 (-1 to 1 range)
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Resample if needed
            if self.telephony_sample_rate != self.ai_sample_rate:
                audio_float = self._resample(
                    audio_float,
                    self.telephony_sample_rate,
                    self.ai_sample_rate
                )

            return audio_float, self.ai_sample_rate

        except Exception as e:
            log.error(f"Error converting telephony→AI audio: {e}")
            # Return silence as fallback
            duration_samples = len(audio_data)
            return np.zeros(duration_samples, dtype=np.float32), self.ai_sample_rate

    def ai_to_telephony(
        self,
        audio_array: np.ndarray,
        format: str = None
    ) -> bytes:
        """
        Convert AI model audio to telephony format.

        Args:
            audio_array: Audio as numpy array (float32, -1 to 1)
            format: Audio format override

        Returns:
            Raw audio bytes for telephony system
        """
        codec = format or self.telephony_codec

        try:
            # Resample if needed
            if self.ai_sample_rate != self.telephony_sample_rate:
                audio_array = self._resample(
                    audio_array,
                    self.ai_sample_rate,
                    self.telephony_sample_rate
                )

            # Convert float32 to int16
            audio_int16 = (audio_array * 32768.0).astype(np.int16)

            # Convert based on codec
            if codec == 'mulaw':
                audio_bytes = self._linear_to_mulaw(audio_int16)
            elif codec == 'alaw':
                audio_bytes = self._linear_to_alaw(audio_int16)
            elif codec == 'linear16':
                audio_bytes = audio_int16.tobytes()
            else:
                raise ValueError(f"Unsupported codec: {codec}")

            return audio_bytes

        except Exception as e:
            log.error(f"Error converting AI→telephony audio: {e}")
            # Return silence
            return b'\xff' * len(audio_array)  # μ-law silence

    def _mulaw_to_linear(self, mulaw_data: bytes) -> np.ndarray:
        """
        Convert μ-law encoded audio to linear PCM.

        Args:
            mulaw_data: μ-law encoded bytes

        Returns:
            Linear PCM as int16 array
        """
        # Use pydub for μ-law decoding
        audio_segment = AudioSegment(
            data=mulaw_data,
            sample_width=1,  # μ-law is 8-bit
            frame_rate=self.telephony_sample_rate,
            channels=1
        )

        # Convert to 16-bit PCM
        audio_segment = audio_segment.set_sample_width(2)

        # Extract as numpy array
        audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)

        return audio_array

    def _linear_to_mulaw(self, linear_data: np.ndarray) -> bytes:
        """
        Convert linear PCM to μ-law encoding.

        Args:
            linear_data: Linear PCM as int16 array

        Returns:
            μ-law encoded bytes
        """
        # Create AudioSegment from array
        audio_segment = AudioSegment(
            data=linear_data.tobytes(),
            sample_width=2,  # 16-bit
            frame_rate=self.telephony_sample_rate,
            channels=1
        )

        # Export as μ-law WAV
        buffer = io.BytesIO()
        audio_segment.export(
            buffer,
            format='wav',
            codec='pcm_mulaw',
            parameters=['-ar', str(self.telephony_sample_rate)]
        )

        # Extract raw μ-law data (skip WAV header)
        buffer.seek(0)
        wav_data = buffer.read()

        # Skip 44-byte WAV header
        mulaw_data = wav_data[44:]

        return mulaw_data

    def _alaw_to_linear(self, alaw_data: bytes) -> np.ndarray:
        """
        Convert A-law encoded audio to linear PCM.
        Similar to μ-law but different encoding table.
        """
        audio_segment = AudioSegment(
            data=alaw_data,
            sample_width=1,
            frame_rate=self.telephony_sample_rate,
            channels=1
        )

        audio_segment = audio_segment.set_sample_width(2)
        audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)

        return audio_array

    def _linear_to_alaw(self, linear_data: np.ndarray) -> bytes:
        """Convert linear PCM to A-law encoding"""
        audio_segment = AudioSegment(
            data=linear_data.tobytes(),
            sample_width=2,
            frame_rate=self.telephony_sample_rate,
            channels=1
        )

        buffer = io.BytesIO()
        audio_segment.export(
            buffer,
            format='wav',
            codec='pcm_alaw',
            parameters=['-ar', str(self.telephony_sample_rate)]
        )

        buffer.seek(0)
        wav_data = buffer.read()
        alaw_data = wav_data[44:]  # Skip WAV header

        return alaw_data

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to different sample rate.

        Args:
            audio: Input audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio
        """
        import librosa

        resampled = librosa.resample(
            audio,
            orig_sr=orig_sr,
            target_sr=target_sr,
            res_type='kaiser_best'  # High quality resampling
        )

        return resampled

    def chunk_audio(
        self,
        audio_data: bytes,
        chunk_duration_ms: int = 20
    ) -> list:
        """
        Split audio into fixed-size chunks for streaming.

        Args:
            audio_data: Audio bytes
            chunk_duration_ms: Chunk duration in milliseconds

        Returns:
            List of audio chunks
        """
        # Calculate bytes per chunk
        bytes_per_second = self.telephony_sample_rate
        if self.telephony_codec in ['mulaw', 'alaw']:
            bytes_per_second *= 1  # 8-bit encoding
        else:
            bytes_per_second *= 2  # 16-bit encoding

        chunk_size = int(bytes_per_second * chunk_duration_ms / 1000)

        # Split into chunks
        chunks = []
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) > 0:
                chunks.append(chunk)

        return chunks

    def create_silence(self, duration_ms: int) -> bytes:
        """
        Create silence in telephony format.

        Args:
            duration_ms: Duration in milliseconds

        Returns:
            Silence audio bytes
        """
        num_samples = int(self.telephony_sample_rate * duration_ms / 1000)

        if self.telephony_codec == 'mulaw':
            # μ-law silence is 0xFF
            return b'\xff' * num_samples
        elif self.telephony_codec == 'alaw':
            # A-law silence is 0xD5
            return b'\xd5' * num_samples
        else:  # linear16
            return b'\x00\x00' * num_samples


class AudioBuffer:
    """
    Circular buffer for audio streaming.
    Handles incoming audio chunks and provides continuous stream.
    """

    def __init__(self, max_duration_seconds: float = 5.0, sample_rate: int = 16000):
        """
        Initialize audio buffer.

        Args:
            max_duration_seconds: Maximum buffer duration
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.size = 0

    def write(self, audio_chunk: np.ndarray):
        """Add audio to buffer"""
        chunk_size = len(audio_chunk)

        # Check overflow
        if self.size + chunk_size > self.max_samples:
            log.warning("Audio buffer overflow, discarding old data")
            self.read_pos = (self.write_pos + chunk_size) % self.max_samples
            self.size = self.max_samples - chunk_size

        # Write to circular buffer
        end_pos = self.write_pos + chunk_size
        if end_pos <= self.max_samples:
            self.buffer[self.write_pos:end_pos] = audio_chunk
        else:
            # Wrap around
            first_part = self.max_samples - self.write_pos
            self.buffer[self.write_pos:] = audio_chunk[:first_part]
            self.buffer[:chunk_size - first_part] = audio_chunk[first_part:]

        self.write_pos = end_pos % self.max_samples
        self.size += chunk_size

    def read(self, num_samples: int) -> np.ndarray:
        """Read audio from buffer"""
        if num_samples > self.size:
            num_samples = self.size

        if num_samples == 0:
            return np.array([], dtype=np.float32)

        # Read from circular buffer
        end_pos = self.read_pos + num_samples
        if end_pos <= self.max_samples:
            result = self.buffer[self.read_pos:end_pos].copy()
        else:
            # Wrap around
            first_part = self.max_samples - self.read_pos
            result = np.concatenate([
                self.buffer[self.read_pos:],
                self.buffer[:num_samples - first_part]
            ])

        self.read_pos = end_pos % self.max_samples
        self.size -= num_samples

        return result

    def available(self) -> int:
        """Get number of available samples"""
        return self.size

    def clear(self):
        """Clear buffer"""
        self.write_pos = 0
        self.read_pos = 0
        self.size = 0

    def get_all(self) -> np.ndarray:
        """Get all buffered audio and clear"""
        audio = self.read(self.size)
        self.clear()
        return audio


# Example usage
if __name__ == "__main__":
    # Test audio bridge
    bridge = AudioBridge(
        telephony_sample_rate=8000,
        telephony_codec='mulaw',
        ai_sample_rate=16000
    )

    # Simulate μ-law audio from telephony (1 second)
    mulaw_data = b'\xff' * 8000

    # Convert to AI format
    ai_audio, sr = bridge.telephony_to_ai(mulaw_data)
    print(f"Converted to AI: {len(ai_audio)} samples @ {sr}Hz")

    # Convert back to telephony
    telephony_audio = bridge.ai_to_telephony(ai_audio)
    print(f"Converted back: {len(telephony_audio)} bytes")

    print("✓ Audio bridge ready")
