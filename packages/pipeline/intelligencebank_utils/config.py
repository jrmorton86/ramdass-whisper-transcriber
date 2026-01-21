"""Configuration for Intelligence Bank utilities."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Settings:
    """Intelligence Bank API settings."""
    
    def __init__(self):
        # Intelligence Bank API settings
        self.ib_platform_url = os.getenv('IB_PLATFORM_URL')
        self.ib_api_email = os.getenv('IB_API_EMAIL')
        self.ib_api_password = os.getenv('IB_API_PASSWORD')
        self.ib_api_v2_url = os.getenv('IB_API_V2_URL')
        self.ib_api_v3_url = os.getenv('IB_API_V3_URL')
        self.ib_client_id = os.getenv('IB_CLIENT_ID')
        
        # Optional cached credentials
        self.ib_session_id = os.getenv('IB_SESSION_ID')
    
    def validate(self):
        """Validate that required settings are present."""
        required = {
            'ib_platform_url': self.ib_platform_url,
            'ib_api_email': self.ib_api_email,
            'ib_api_password': self.ib_api_password,
            'ib_client_id': self.ib_client_id,
        }
        
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing required Intelligence Bank configuration: {', '.join(missing)}")


# Global settings instance
settings = Settings()
