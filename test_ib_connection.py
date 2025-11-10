"""Test Intelligence Bank connection."""
import asyncio
from intelligencebank_utils.ib_client import ib_client
from intelligencebank_utils.config import settings


async def test_connection():
    """Test connection to Intelligence Bank."""
    print("Testing Intelligence Bank connection...")
    print(f"Platform URL: {settings.ib_platform_url}")
    print(f"Email: {settings.ib_api_email}")
    
    try:
        # Validate configuration
        settings.validate()
        print("✓ Configuration validated")
        
        # Authenticate
        await ib_client.ensure_authenticated()
        print("✓ Authentication successful!")
        
        # Display connection details (essential credentials only)
        print(f"\nConnection Details:")
        print(f"  Session ID (SUPER ESSENTIAL): {ib_client.session_id[:20]}..." if ib_client.session_id else "  Session ID: None")
        print(f"  Client ID: {ib_client.client_id}")
        print(f"  API V2 URL: {ib_client.api_v2_url}")
        print(f"  API V3 URL: {ib_client.api_v3_url}")
        
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_connection())
    if success:
        print("\n✓ Successfully connected to Intelligence Bank!")
    else:
        print("\n✗ Failed to connect to Intelligence Bank")
