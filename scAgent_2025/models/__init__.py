"""
Model interface and API utilities for scAgent.
"""

from .qwen_client import (
    QwenClient, 
    get_qwen_client, 
    create_analysis_prompt,
    create_eqtl_evaluation_prompt,
    create_data_cleaning_prompt
)
from .base import BaseModel, ModelResponse
from .client import ModelClient, get_model_client

__all__ = [
    "QwenClient",
    "get_qwen_client",
    "create_analysis_prompt",
    "create_eqtl_evaluation_prompt",
    "create_data_cleaning_prompt",
    "BaseModel", 
    "ModelResponse",
    "ModelClient",
    "get_model_client"
] 