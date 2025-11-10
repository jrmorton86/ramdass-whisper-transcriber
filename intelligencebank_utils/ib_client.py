"""Intelligence Bank API client."""
import httpx
from typing import Optional
from .config import settings


class IBClient:
    """Client for Intelligence Bank API."""
    
    def __init__(self):
        self.platform_url = settings.ib_platform_url
        self.email = settings.ib_api_email
        self.password = settings.ib_api_password
        self.api_v2_url = settings.ib_api_v2_url
        self.api_v3_url = settings.ib_api_v3_url
        # Essential credentials
        self.session_id = settings.ib_session_id  # SUPER ESSENTIAL
        self.client_id = None
    
    async def ensure_authenticated(self):
        """Ensure we have valid API credentials, refresh if needed."""
        if self.session_id:
            return  # Already authenticated
        
        # Step 1: Get API V2 address
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://{self.platform_url}/v1/auth/app/getYapiAddress",
                timeout=30.0
            )
            resp.raise_for_status()
            data = resp.json()
            self.api_v2_url = data.get('content')
        
        # Step 2: Login
        async with httpx.AsyncClient() as client:
            login_data = {
                'p70': self.email,
                'p80': self.password,
                'p90': self.platform_url
            }
            resp = await client.post(
                f"{self.api_v2_url}/webapp/1.0/login",
                data=login_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=30.0
            )
            resp.raise_for_status()
            result = resp.json()
            
            # Extract essential credentials
            self.session_id = result.get('sid')  # SUPER ESSENTIAL
            self.api_v3_url = result.get('apiV3url')
            self.client_id = result.get('clientid')
    
    async def get_authenticated_headers(self) -> dict:
        """Get headers with authentication (session ID is the essential credential)."""
        await self.ensure_authenticated()
        return {
            'sid': self.session_id  # SUPER ESSENTIAL - this is what authenticates requests
        }
    
    def get_v3_base_url(self) -> str:
        """Get V3 API base URL with client ID."""
        if not self.client_id:
            # Fallback to stored client ID if not set
            self.client_id = 'vJgXP3'
        return f"{self.api_v3_url}/api/3.0.0/{self.client_id}"
    
    async def fetch_asset_stream(self, url: str):
        """
        Fetch an asset stream from Intelligence Bank.
        
        Returns:
            httpx.Response with streaming enabled
        """
        headers = await self.get_authenticated_headers()
        client = httpx.AsyncClient(timeout=300.0)
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response, client  # Client must be kept alive for streaming


# Singleton instance
ib_client = IBClient()
