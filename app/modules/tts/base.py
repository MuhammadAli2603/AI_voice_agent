"""
Abstract base class for Text-to-Speech modules
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Union


class BaseTTS(ABC):
    """
    Abstract base class for Text-to-Speech implementations
    Ensures all TTS modules follow the same interface
    """
    
    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize TTS module
        
        Args:
            model_name: Name/path of the model
            device: Computing device (cpu, cuda, mps)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        
    @abstractmethod
    def load_model(self):
        """Load the TTS model"""
        pass
    
    @abstractmethod
    def synthesize(
        self,
        text: str,
        language: str = "en",
        speed: float = 1.0
    ) -> np.ndarray:
        """
        Synthesize text to speech
        
        Args:
            text: Text to synthesize
            language: Language code
            speed: Speech speed multiplier
            
        Returns:
            Audio array
        """
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        pass