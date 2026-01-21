"""Download an asset from Intelligence Bank by asset ID."""
import asyncio
import argparse
import re
from pathlib import Path
from typing import Optional
import httpx

from .ib_client import ib_client


def normalize_uuid(asset_id: str) -> str:
    """
    Normalize UUID to remove dashes for API calls.
    
    Args:
        asset_id: UUID with or without dashes (e.g., 'fca0a091-a6d0-41d5-8a26-1932a9aa6de1' or 'fca0a091a6d041d58a261932a9aa6de1')
    
    Returns:
        UUID without dashes (e.g., 'fca0a091a6d041d58a261932a9aa6de1')
    
    Raises:
        ValueError: If the input is not a valid UUID format
    """
    # Remove dashes if present
    normalized = asset_id.replace('-', '')
    
    # Validate it's a valid UUID length (32 hex characters)
    if len(normalized) != 32:
        raise ValueError(f"Invalid UUID format: expected 32 characters, got {len(normalized)}")
    
    # Validate it's all hex characters
    if not re.match(r'^[0-9a-fA-F]{32}$', normalized):
        raise ValueError(f"Invalid UUID format: must contain only hexadecimal characters")
    
    return normalized


async def get_resource_by_uuid(uuid: str) -> Optional[dict]:
    """
    Get resource metadata from Intelligence Bank API.
    
    Args:
        uuid: Resource UUID without dashes
    
    Returns:
        Resource metadata dict or None
    """
    await ib_client.ensure_authenticated()
    headers = await ib_client.get_authenticated_headers()
    headers['Content-Type'] = 'application/json'
    
    # Build URL: POST {{apiV3url}}/api/json
    url = f"{ib_client.api_v3_url}/api/json"
    
    # Build POST payload with wrapped_conditions
    payload = {
        "method": "GET",
        "version": "3.0.0",
        "client": ib_client.client_id or "vJgXP3",
        "table": "resource.limit(1)",
        "query_params": {
            "searchParams": {
                "ib_folder_s": "",
                "isSearching": True,
                "wrapped_conditions": [
                    [
                        {
                            "field": "ib_uuid",
                            "value": uuid,
                            "union": "and",
                            "type": "text",
                            "op": "equals"
                        }
                    ]
                ]
            }
        }
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # IB V3 API returns: {"response": {"count": N, "rows": [...]}}
        if isinstance(data, dict) and 'response' in data:
            resp = data['response']
            rows = resp.get('rows', [])
            if len(rows) > 0:
                return rows[0]
        return None


async def get_download_url(asset_id: str) -> str:
    """
    Get the download URL for an asset.
    
    Args:
        asset_id: Asset UUID (with or without dashes)
    
    Returns:
        Download URL for the asset
    """
    # Normalize UUID (remove dashes)
    normalized_id = normalize_uuid(asset_id)
    
    # Get the resource metadata which contains the download URL
    resource = await get_resource_by_uuid(normalized_id)
    if not resource:
        raise ValueError(f"Asset not found: {asset_id}")
    
    # Extract download URL from resource - try various fields
    # publicShareLink is the CDN URL for downloading the original file
    download_url = (
        resource.get('publicShareLink') or 
        resource.get('downloadUrl') or 
        resource.get('url')
    )
    if not download_url:
        raise ValueError(f"No download URL found for asset: {asset_id}")
    
    return download_url


async def download_asset(asset_id: str, output_path: Optional[str] = None) -> Path:
    """
    Download an asset from Intelligence Bank.
    
    Args:
        asset_id: Asset UUID (with or without dashes)
        output_path: Optional custom output path. If not provided, uses asset ID as filename.
    
    Returns:
        Path to the downloaded file
    """
    print(f"Downloading asset: {asset_id}")
    
    # Get download URL
    download_url = await get_download_url(asset_id)
    print(f"Download URL: {download_url}")
    
    # Get authenticated headers
    headers = await ib_client.get_authenticated_headers()
    
    # Download the file
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.get(download_url, headers=headers, follow_redirects=True)
        response.raise_for_status()
        
        # Determine output filename
        if output_path:
            output_file = Path(output_path)
        else:
            # Try to get filename from Content-Disposition header
            content_disposition = response.headers.get('content-disposition', '')
            filename_match = re.search(r'filename="?([^"]+)"?', content_disposition)
            
            if filename_match:
                filename = filename_match.group(1)
            else:
                # Extract extension from Content-Type if available
                content_type = response.headers.get('content-type', '')
                extension = ''
                if 'pdf' in content_type:
                    extension = '.pdf'
                elif 'image' in content_type:
                    if 'jpeg' in content_type or 'jpg' in content_type:
                        extension = '.jpg'
                    elif 'png' in content_type:
                        extension = '.png'
                elif 'audio' in content_type:
                    if 'mpeg' in content_type or 'mp3' in content_type:
                        extension = '.mp3'
                    elif 'wav' in content_type:
                        extension = '.wav'
                elif 'video' in content_type:
                    if 'mp4' in content_type:
                        extension = '.mp4'
                
                # Use asset ID as filename with appropriate extension
                normalized_id = normalize_uuid(asset_id)
                filename = f"{normalized_id}{extension}"
            
            # Save to downloads directory
            tmp_dir = Path(__file__).parent.parent / 'tmp'
            tmp_dir.mkdir(exist_ok=True)
            output_file = tmp_dir / filename
        
        # Write file
        output_file.write_bytes(response.content)
        print(f"✓ Downloaded to: {output_file}")
        print(f"  File size: {len(response.content):,} bytes")
        print(f"  Content type: {response.headers.get('content-type', 'unknown')}")
        
        return output_file


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Download an asset from Intelligence Bank')
    parser.add_argument('asset_id', help='Asset UUID (with or without dashes)')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    
    args = parser.parse_args()
    
    try:
        output_file = await download_asset(args.asset_id, args.output)
        print(f"\n✓ Success! Asset downloaded to: {output_file}")
        return 0
    except ValueError as e:
        print(f"✗ Error: {e}")
        return 1
    except Exception as e:
        print(f"✗ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
