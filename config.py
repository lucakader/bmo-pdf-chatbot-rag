"""
Configuration settings for the PDF chatbot application.
"""
import os
from typing import Dict, Any, Optional, List, Tuple

# Vector store settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
VECTOR_DIMENSION = 1536

# Retrieval settings
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
DEFAULT_SIMILARITY_TOP_K = int(os.getenv("DEFAULT_SIMILARITY_TOP_K", "5"))
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"

# LLM settings
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")
DEFAULT_LLM_TEMPERATURE = float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.0"))
HALLUCINATION_CHECK_ENABLED = os.getenv("HALLUCINATION_CHECK_ENABLED", "true").lower() == "true"

# Metrics settings
METRICS_PORT = int(os.getenv("METRICS_PORT", "8099"))
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"

# Performance settings
CACHE_EMBEDDINGS = os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true"
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))

# Required environment variables
REQUIRED_ENV_VARS = {
    "OPENAI_API_KEY": "OpenAI API key for embeddings and LLM",
    "PINECONE_API_KEY": "Pinecone API key for vector database",
    "PINECONE_ENVIRONMENT": "Pinecone environment (e.g., us-east1-gcp)"
}

def validate_environment() -> Tuple[bool, List[str]]:
    """
    Validate that all required environment variables are set.
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, missing_vars)
          - is_valid: True if all required variables are set, False otherwise
          - missing_vars: List of missing environment variables
    """
    missing = []
    for var, description in REQUIRED_ENV_VARS.items():
        if not os.getenv(var):
            missing.append(f"{var} ({description})")
    
    return len(missing) == 0, missing 