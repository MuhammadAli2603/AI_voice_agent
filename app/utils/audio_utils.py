"""
Audio processing utilities
"""
import io
import numpy as np
import librosa
import soundfile as sf
from typing import Union, Tuple
from app.utils.logger import log


def load_audio(
    audio_input: Union[str, bytes, io.BytesIO],
    sample_rate: int = 16000
) -> Tuple[np.ndarray, int]:
    """
    Load audio from various sources and resample
    
    Args:
        audio_input: Audio file path, bytes, or BytesIO object
        sample_rate: Target sample rate
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        if isinstance(audio_input, str):
            # Load from file path
            audio, sr = librosa.load(audio_input, sr=sample_rate)
        elif isinstance(audio_input, bytes):
            # Load from bytes
            audio, sr = sf.read(io.BytesIO(audio_input))
            if sr != sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                sr = sample_rate
        elif isinstance(audio_input, io.BytesIO):
            # Load from BytesIO
            audio, sr = sf.read(audio_input)
            if sr != sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
                sr = sample_rate
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
        
        log.info(f"Audio loaded successfully: shape={audio.shape}, sr={sr}")
        return audio, sr
        
    except Exception as e:
        log.error(f"Error loading audio: {e}")
        raise


def save_audio(
    audio: np.ndarray,
    output_path: str,
    sample_rate: int = 22050
) -> None:
    """
    Save audio array to file
    
    Args:
        audio: Audio array
        output_path: Output file path
        sample_rate: Sample rate
    """
    try:
        sf.write(output_path, audio, sample_rate)
        log.info(f"Audio saved to {output_path}")
    except Exception as e:
        log.error(f"Error saving audio: {e}")
        raise


def audio_to_bytes(
    audio: np.ndarray,
    sample_rate: int = 22050,
    format: str = "WAV"
) -> bytes:
    """
    Convert audio array to bytes
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        format: Audio format (WAV, MP3, etc.)
        
    Returns:
        Audio as bytes
    """
    try:
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format=format)
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        log.error(f"Error converting audio to bytes: {e}")
        raise


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range
    
    Args:
        audio: Audio array
        
    Returns:
        Normalized audio
    """
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio / max_val
    return audio


def trim_silence(
    audio: np.ndarray,
    sample_rate: int = 16000,
    top_db: int = 20
) -> np.ndarray:
    """
    Trim silence from audio
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        top_db: Threshold in dB
        
    Returns:
        Trimmed audio
    """
    try:
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    except Exception as e:
        log.error(f"Error trimming silence: {e}")
        return audio