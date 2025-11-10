"""Configuration management for database navigator."""
import os
import json
import boto3
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Settings:
    """Application settings loaded from environment variables."""
    
    def __init__(self):
        # Database settings
        self.db_host = os.getenv('DATABASE_HOST')
        self.db_port = int(os.getenv('DATABASE_PORT', 5432))
        self.db_name = os.getenv('DATABASE_NAME')
        self.db_user = os.getenv('DATABASE_USER')
        
        # AWS settings
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.use_secrets_manager = os.getenv('USE_SECRETS_MANAGER', 'false').lower() == 'true'
        self.db_secret_name = os.getenv('DB_SECRET_NAME')
        
        # S3 settings
        self.ingest_s3_bucket = os.getenv('INGEST_S3_BUCKET')
        
        # Redis settings
        self.redis_url = os.getenv('REDIS_URL')
        
        # Cache the password
        self._db_password: Optional[str] = None
    
    @property
    def db_password(self) -> str:
        """Get database password from Secrets Manager or environment."""
        if self._db_password is not None:
            return self._db_password
        
        if self.use_secrets_manager and self.db_secret_name:
            self._db_password = self._get_secret_from_aws()
        else:
            # Fallback to environment variable
            self._db_password = os.getenv('DATABASE_PASSWORD', '')
        
        return self._db_password
    
    def _get_secret_from_aws(self) -> str:
        """Retrieve database password from AWS Secrets Manager."""
        try:
            client = boto3.client(
                service_name='secretsmanager',
                region_name=self.aws_region
            )
            
            response = client.get_secret_value(SecretId=self.db_secret_name)
            
            # Parse the secret (it might be JSON or plain text)
            if 'SecretString' in response:
                secret = response['SecretString']
                try:
                    # Try to parse as JSON first
                    secret_dict = json.loads(secret)
                    # Common keys for RDS secrets
                    return secret_dict.get('password') or secret_dict.get('Password') or secret
                except json.JSONDecodeError:
                    # If not JSON, return as-is
                    return secret
            else:
                raise ValueError("Secret not found in expected format")
                
        except Exception as e:
            raise Exception(f"Failed to retrieve secret from AWS Secrets Manager: {str(e)}")
    
    def validate(self):
        """Validate that required settings are present."""
        required = {
            'db_host': self.db_host,
            'db_port': self.db_port,
            'db_name': self.db_name,
            'db_user': self.db_user,
        }
        
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
        
        # Check password availability
        if self.use_secrets_manager and not self.db_secret_name:
            raise ValueError("USE_SECRETS_MANAGER is true but DB_SECRET_NAME is not set")


# Global settings instance
settings = Settings()
