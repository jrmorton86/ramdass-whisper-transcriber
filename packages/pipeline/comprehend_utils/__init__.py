"""
AWS Comprehend and Embedding Generation Utilities

This module provides utilities for:
- Analyzing transcripts with AWS Comprehend
- Generating embeddings with Amazon Titan
- Storing results in PostgreSQL database
"""

from .comprehend import (
    analyze_transcript_with_comprehend,
    generate_embedding,
    store_comprehend_analysis,
    store_embeddings,
)

from .config import (
    AWS_REGION,
    get_bedrock_runtime_client,
    get_comprehend_client,
)

__all__ = [
    # Comprehend functions
    'analyze_transcript_with_comprehend',
    'generate_embedding',
    'store_comprehend_analysis',
    'store_embeddings',
    
    # Config
    'AWS_REGION',
    'get_bedrock_runtime_client',
    'get_comprehend_client',
]
