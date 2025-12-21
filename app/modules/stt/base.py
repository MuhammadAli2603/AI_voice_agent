"""
Abstract base class for Speech-to-Text modules
"""
from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class BaseSTT(ABC):
    """
    Abstract base class for Speech-to-Text implementations
    Ensures all STT modules follow the same interface
    """
    
    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize STT module
        
        Args:
            model_name: Name/path of the model
            device: Computing device (cpu, cuda, mps)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        
    @abstractmethod
    def load_model(self):
        """Load the STT model"""
        pass
    
    @abstractmethod
    def transcribe(
        self,
        audio: Union[np.ndarray, str],
        language: str = "en"
    ) -> dict:
        """
        Transcribe audio to text
        
        Args:
            audio: Audio array or file path
            language: Language code
            
        Returns:
            Dictionary with 'text' and optional 'confidence'
        """
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        pass