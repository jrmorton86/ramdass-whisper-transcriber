"""
AWS Configuration for Comprehend and Bedrock Services
"""

import os
import boto3
from typing import Optional, Any

# AWS Region
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# Bedrock runtime client (cached)
_bedrock_runtime_client: Optional[Any] = None


def get_bedrock_runtime_client() -> Any:
    """
    Get or create a cached Bedrock Runtime client.
    
    Returns:
        boto3 Bedrock Runtime client
    """
    global _bedrock_runtime_client
    
    if _bedrock_runtime_client is None:
        _bedrock_runtime_client = boto3.client(
            'bedrock-runtime',
            region_name=AWS_REGION
        )
    
    return _bedrock_runtime_client


def get_comprehend_client() -> Any:
    """
    Get an AWS Comprehend client.
    
    Returns:
        boto3 Comprehend client
    """
    return boto3.client('comprehend', region_name=AWS_REGION)
