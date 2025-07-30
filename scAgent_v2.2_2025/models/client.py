"""
Model client interface for scAgent.
"""

from .qwen_client import QwenClient, get_qwen_client

# Create ModelClient as an alias for QwenClient
ModelClient = QwenClient

def get_model_client(**kwargs):
    """Get a model client instance."""
    return get_qwen_client(**kwargs)

__all__ = ["ModelClient", "get_model_client"] 