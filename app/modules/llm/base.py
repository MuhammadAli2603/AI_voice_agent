"""
Abstract base class for LLM modules
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseLLM(ABC):
    """
    Abstract base class for Language Model implementations
    Ensures all LLM modules follow the same interface
    """
    
    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize LLM module
        
        Args:
            model_name: Name/path of the model
            device: Computing device (cpu, cuda, mps)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
    @abstractmethod
    def load_model(self):
        """Load the LLM model"""
        pass
    
    @abstractmethod
    def generate_response(
        self,
        message: str,
        conversation_history: Optional[List[Dict]] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate response to user message
        
        Args:
            message: User message
            conversation_history: Previous conversation
            max_length: Maximum response length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated response
        """
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        pass