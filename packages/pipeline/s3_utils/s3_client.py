"""S3 streaming upload utilities."""
import sys
from pathlib import Path
import boto3
from typing import BinaryIO, Optional, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from s3_utils.config import settings

s3_client = boto3.client('s3', region_name=settings.aws_region)


def upload_stream(bucket: str, key: str, stream: BinaryIO, content_type: Optional[str] = None, metadata: Optional[Dict] = None):
    """Upload a stream to S3."""
    extra_args = {}
    if content_type:
        extra_args['ContentType'] = content_type
    if metadata:
        extra_args['Metadata'] = {k: str(v) for k, v in metadata.items()}
    
    s3_client.upload_fileobj(stream, bucket, key, ExtraArgs=extra_args)
    return {'bucket': bucket, 'key': key}


def download_stream(bucket: str, key: str) -> bytes:
    """Download object from S3 as bytes."""
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response['Body'].read()


def delete_object(bucket: str, key: str):
    """Delete object from S3."""
    s3_client.delete_object(Bucket=bucket, Key=key)
