"""
Base model interface for scAgent.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelResponse:
    """Response from a model API call."""
    content: str
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

class BaseModel(ABC):
    """Base class for model interfaces."""
    
    def __init__(self, model_name: str, api_base: str, **kwargs):
        self.model_name = model_name
        self.api_base = api_base
        self.config = kwargs
        
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 32000,
        **kwargs
    ) -> ModelResponse:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 32000,
        **kwargs
    ) -> ModelResponse:
        """Have a chat conversation with the model."""
        pass
    
    def _prepare_request(self, **kwargs) -> Dict[str, Any]:
        """Prepare request parameters."""
        request_params = {
            "model": self.model_name,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 32000),
        }
        
        # Add additional parameters
        for key, value in kwargs.items():
            if key not in request_params and value is not None:
                request_params[key] = value
                
        return request_params
    
    def _handle_response(self, response: Dict[str, Any]) -> ModelResponse:
        """Handle and parse model response."""
        try:
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = response.get("usage", {})
            model = response.get("model", self.model_name)
            finish_reason = response.get("choices", [{}])[0].get("finish_reason", "")
            
            return ModelResponse(
                content=content,
                usage=usage,
                model=model,
                finish_reason=finish_reason,
                raw_response=response
            )
        except Exception as e:
            logger.error(f"Error parsing model response: {e}")
            return ModelResponse(
                content="",
                raw_response=response
            ) 