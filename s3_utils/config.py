"""Configuration for S3 utilities."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Settings:
    """S3 settings."""
    
    def __init__(self):
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.s3_bucket = os.getenv('S3_BUCKET', 'dam-ramdass-io-assets')


# Global settings instance
settings = Settings()
