"""
Receptionist LLM implementation using Hugging Face Inference API
No local model download required!
Integrated with Knowledge Base RAG system for context-aware responses.
"""
import requests
from typing import List, Dict, Optional

from app.modules.llm.base import BaseLLM
from app.utils.logger import log


class ReceptionistLLM(BaseLLM):
    """
    Friendly receptionist LLM using Hugging Face Inference API
    Uses cloud-based inference - no local models needed!
    """
    
    # Receptionist system prompt
    RECEPTIONIST_SYSTEM = """You are a friendly and professional receptionist AI assistant. Your role is to:
- Greet visitors warmly and make them feel welcome
- Listen carefully to their needs and respond helpfully
- Be polite, patient, and understanding
- Provide clear and concise information
- Maintain a positive and professional tone
- Show empathy and genuine care for helping people

Keep your responses natural, conversational, and friendly. Avoid being overly formal or robotic.
Be brief and to the point - aim for 1-3 sentences unless more detail is specifically requested."""
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        api_key: str = "",
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        kb_service_url: str = "http://localhost:8001"
    ):
        """
        Initialize Receptionist LLM with API and Knowledge Base integration

        Args:
            model_name: HuggingFace model ID
            api_key: Hugging Face API key
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            kb_service_url: Knowledge Base service URL
        """
        super().__init__(model_name, device="api")
        self.api_key = api_key
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.conversation_history = []
        self.kb_service_url = kb_service_url
        self.load_model()
        
    def load_model(self):
        """Verify API access"""
        try:
            log.info(f"Initializing API connection for: {self.model_name}")
            
            if not self.api_key or self.api_key == "your_huggingface_api_key_here":
                log.warning("⚠️  No valid Hugging Face API key provided!")
                log.warning("Get your API key from: https://huggingface.co/settings/tokens")
            
            # Mark as loaded
            self.model = "api"
            self.tokenizer = "api"
            log.info("✓ LLM API initialized successfully")
            
        except Exception as e:
            log.error(f"Error initializing LLM API: {e}")
            raise
    
    def _query_knowledge_base(
        self,
        query: str,
        company_id: str,
        top_k: int = 3
    ) -> Optional[str]:
        """
        Query the knowledge base for relevant context.

        Args:
            query: User query
            company_id: Company identifier
            top_k: Number of results to retrieve

        Returns:
            Formatted context string or None if KB unavailable
        """
        try:
            log.info(f"Querying knowledge base for company: {company_id}")

            response = requests.post(
                f"{self.kb_service_url}/api/v1/context",
                json={
                    "company_id": company_id,
                    "query": query,
                    "top_k": top_k,
                    "include_metadata": False
                },
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                context = result.get("context", "")
                confidence = result.get("confidence_score", 0.0)

                log.info(f"KB context retrieved (confidence: {confidence:.2f})")
                return context if confidence > 0.3 else None
            else:
                log.warning(f"KB query failed: {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            log.warning("Knowledge base query timed out")
            return None
        except Exception as e:
            log.warning(f"Knowledge base unavailable: {e}")
            return None

    def generate_response(
        self,
        message: str,
        conversation_history: Optional[List[Dict]] = None,
        max_length: int = None,
        temperature: float = None,
        top_p: float = None,
        company_id: Optional[str] = None
    ) -> str:
        """
        Generate friendly receptionist response using API with KB context

        Args:
            message: User message
            conversation_history: Previous conversation (optional)
            max_length: Maximum response length
            temperature: Temperature
            top_p: Top-p
            company_id: Company ID for knowledge base context (optional)

        Returns:
            Generated response
        """
        try:
            # Use provided values or defaults
            max_new_tokens = max_length or 150
            temp = temperature or self.temperature
            top_p_val = top_p or self.top_p

            # Query knowledge base if company_id provided
            kb_context = None
            if company_id:
                kb_context = self._query_knowledge_base(message, company_id)

            # Build prompt with system message, KB context, and conversation
            prompt = self._build_prompt(message, kb_context=kb_context)
            
            log.info(f"Sending request to LLM API: '{message[:50]}...'")
            
            # Prepare API payload
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temp,
                    "top_p": top_p_val,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            # Make API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract generated text from response
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict) and "generated_text" in result[0]:
                        generated = result[0]["generated_text"]
                    else:
                        generated = str(result[0])
                elif isinstance(result, dict) and "generated_text" in result:
                    generated = result["generated_text"]
                else:
                    generated = str(result)
                
                # Clean response
                response_text = self._clean_response(generated)
                
                # Add to conversation history
                self.conversation_history.append({"user": message, "assistant": response_text})
                
                log.info(f"✓ LLM Response: {response_text}")
                return response_text
            
            elif response.status_code == 503:
                log.warning("Model is loading... This may take a minute on first use")
                return "I'm getting ready to help you. Please try again in a moment."
            
            else:
                error_msg = response.json().get("error", response.text)
                log.error(f"API Error ({response.status_code}): {error_msg}")
                return "I apologize, but I'm having trouble processing your request right now. How else may I assist you?"
            
        except requests.exceptions.Timeout:
            log.error("API request timed out")
            return "I apologize for the delay. Could you please repeat that?"
        except Exception as e:
            log.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble right now. How else may I help you?"
    
    def _build_prompt(self, message: str, kb_context: Optional[str] = None) -> str:
        """
        Build prompt with system message, KB context, and conversation history

        Args:
            message: Current user message
            kb_context: Knowledge base context (optional)

        Returns:
            Formatted prompt
        """
        # Start with system message
        prompt = f"{self.RECEPTIONIST_SYSTEM}\n\n"

        # Add knowledge base context if available
        if kb_context:
            prompt += "Relevant Information from Knowledge Base:\n"
            prompt += f"{kb_context}\n\n"
            prompt += "Use the above information to answer the user's question accurately. "
            prompt += "If the information doesn't contain the answer, politely say you don't have that information.\n\n"

        # Add recent conversation history (last 3 turns)
        recent_history = self.conversation_history[-3:] if self.conversation_history else []
        for turn in recent_history:
            prompt += f"User: {turn['user']}\n"
            prompt += f"Assistant: {turn['assistant']}\n\n"

        # Add current message
        prompt += f"User: {message}\nAssistant:"

        return prompt
    
    def _clean_response(self, response: str) -> str:
        """
        Clean and normalize the response
        
        Args:
            response: Raw response from model
            
        Returns:
            Cleaned response
        """
        # Remove potential unwanted prefixes
        response = response.strip()
        
        # Remove common prefixes
        prefixes = ["Receptionist:", "Assistant:", "AI:", "Bot:"]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Remove "User:" if accidentally generated
        if "User:" in response:
            response = response.split("User:")[0].strip()
        
        # Ensure proper ending punctuation
        if response and response[-1] not in ['.', '!', '?']:
            response += '.'
        
        # Limit length (safety check)
        sentences = response.split('. ')
        if len(sentences) > 4:
            response = '. '.join(sentences[:4]) + '.'
        
        return response
    
    def reset_conversation(self):
        """Reset conversation memory"""
        self.conversation_history = []
        log.info("Conversation memory reset")
    
    def is_loaded(self) -> bool:
        """Check if API is initialized"""
        return self.model == "api" and self.tokenizer == "api"