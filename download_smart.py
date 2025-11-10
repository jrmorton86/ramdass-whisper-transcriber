"""Download asset from S3 or Intelligence Bank with automatic fallback."""
import asyncio
import argparse
import os
import re
from pathlib import Path
from typing import Optional, Tuple
import boto3
from botocore.exceptions import ClientError

from intelligencebank_utils.download_asset import normalize_uuid, get_resource_by_uuid

# S3 settings
S3_BUCKET = os.getenv('S3_BUCKET', 'dam-ramdass-io-assets')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')


def add_uuid_hyphens(uuid_no_dashes: str) -> str:
    """
    Add hyphens to UUID string.
    
    Args:
        uuid_no_dashes: UUID without dashes (e.g., 'fca0a091a6d041d58a261932a9aa6de1')
    
    Returns:
        UUID with hyphens (e.g., 'fca0a091-a6d0-41d5-8a26-1932a9aa6de1')
    """
    if len(uuid_no_dashes) != 32:
        raise ValueError(f"Invalid UUID length: {len(uuid_no_dashes)}")
    
    return f"{uuid_no_dashes[0:8]}-{uuid_no_dashes[8:12]}-{uuid_no_dashes[12:16]}-{uuid_no_dashes[16:20]}-{uuid_no_dashes[20:32]}"


def get_asset_type_from_mime(mime_type: str) -> str:
    """
    Get asset type folder based on MIME type.
    
    Args:
        mime_type: MIME type string (e.g., 'audio/mpeg', 'application/pdf')
    
    Returns:
        Asset type folder name: 'audio', 'documents', 'videos', or 'photo'
    """
    if not mime_type:
        return 'documents'
    
    mime_lower = mime_type.lower()
    
    if mime_lower.startswith('audio/'):
        return 'audio'
    elif mime_lower.startswith('video/'):
        return 'videos'
    elif mime_lower.startswith('image/'):
        return 'photo'
    else:
        return 'documents'


def get_mime_category(mime_type: str) -> str:
    """
    Get MIME category (audio, video, image, etc.) from full MIME type.
    
    Args:
        mime_type: Full MIME type (e.g., 'audio/mpeg', 'image/jpeg')
    
    Returns:
        MIME category (e.g., 'audio', 'video', 'image', 'application')
    """
    if not mime_type or '/' not in mime_type:
        return 'application'
    return mime_type.split('/')[0].lower()


def get_mime_category_from_extension(filename: str) -> Optional[str]:
    """
    Guess MIME category from file extension.
    
    Args:
        filename: File name or path
    
    Returns:
        MIME category or None
    """
    ext = Path(filename).suffix.lower()
    
    audio_exts = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.aiff'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.tiff', '.ico'}
    
    if ext in audio_exts:
        return 'audio'
    elif ext in video_exts:
        return 'video'
    elif ext in image_exts:
        return 'image'
    return None


async def check_s3_for_asset(uuid: str, expected_mime_type: Optional[str] = None, asset_type: Optional[str] = None) -> Optional[list]:
    """
    Check if asset exists in S3 bucket and return all files matching the MIME type.
    
    Args:
        uuid: Asset UUID (with or without dashes)
        expected_mime_type: Expected MIME type from IB (e.g., 'audio/x-wav')
        asset_type: Optional asset type ('audio', 'documents', 'videos', 'photo')
                   If not provided, will try all types
    
    Returns:
        List of tuples [(s3_key, content_type), ...] if found, None otherwise
    """
    # Normalize and add hyphens for S3 path
    normalized = normalize_uuid(uuid)
    uuid_with_hyphens = add_uuid_hyphens(normalized)
    
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    bucket = S3_BUCKET
    
    # Asset types to check
    types_to_check = [asset_type] if asset_type else ['audio', 'documents', 'videos', 'photo']
    
    # Determine expected MIME category for filtering
    expected_category = get_mime_category(expected_mime_type) if expected_mime_type else None
    
    for atype in types_to_check:
        # List objects under this UUID folder
        prefix = f"{atype}/{uuid_with_hyphens}/"
        
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=50  # Increased to handle more files
            )
            
            if 'Contents' in response and len(response['Contents']) > 0:
                # Filter out system/temp files
                valid_files = [
                    obj for obj in response['Contents']
                    if not obj['Key'].endswith('.write_access_check_file.temp')
                    and not obj['Key'].endswith('/')  # Skip folder markers
                ]
                
                if valid_files:
                    # Get content types for all files and filter by MIME category
                    matching_files = []
                    companion_files = []  # JSON, SRT, TXT, etc.
                    
                    # Common companion file extensions
                    companion_extensions = {'.json', '.srt', '.txt', '.vtt', '.xml', '.metadata'}
                    
                    for obj in valid_files:
                        s3_key = obj['Key']
                        file_ext = Path(s3_key).suffix.lower()
                        
                        # Get content type
                        try:
                            head_response = s3_client.head_object(Bucket=bucket, Key=s3_key)
                            content_type = head_response.get('ContentType', 'application/octet-stream')
                            
                            # If we have an expected category, filter by it
                            if expected_category:
                                file_category = get_mime_category(content_type)
                                
                                # If MIME type is generic (binary/octet-stream), use extension
                                if content_type in ['binary/octet-stream', 'application/octet-stream']:
                                    file_category = get_mime_category_from_extension(s3_key) or 'application'
                                
                                # Primary file matches the MIME category
                                if file_category == expected_category:
                                    matching_files.append((s3_key, content_type))
                                # Companion files are common metadata/transcription files
                                elif file_ext in companion_extensions:
                                    companion_files.append((s3_key, content_type))
                            else:
                                # No filter, include all valid files
                                matching_files.append((s3_key, content_type))
                        except ClientError:
                            continue
                    
                    # Combine primary files with companion files
                    all_files = matching_files + companion_files
                    
                    if all_files:
                        print(f"✓ Found {len(all_files)} file(s) in S3 (MIME: {expected_mime_type}):")
                        if matching_files:
                            print(f"  Primary files ({len(matching_files)}):")
                            for s3_key, ct in matching_files:
                                print(f"    - s3://{bucket}/{s3_key} ({ct})")
                        if companion_files:
                            print(f"  Companion files ({len(companion_files)}):")
                            for s3_key, ct in companion_files:
                                print(f"    - s3://{bucket}/{s3_key} ({ct})")
                        return all_files
                
        except ClientError as e:
            # Continue to next type
            continue
    
    return None


async def download_from_s3(s3_key: str, output_path: Optional[Path] = None, uuid_with_hyphens: Optional[str] = None) -> Path:
    """
    Download single file from S3.
    
    Args:
        s3_key: S3 object key
        output_path: Optional output path
        uuid_with_hyphens: UUID with hyphens for directory naming
    
    Returns:
        Path to downloaded file
    """
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    bucket = S3_BUCKET
    
    # Determine output filename
    if output_path:
        output_file = output_path
    else:
        # Use filename from S3 key
        filename = Path(s3_key).name
        # Download to /tmp/{uuid-with-hyphens}/
        if uuid_with_hyphens:
            downloads_dir = Path(__file__).parent / 'tmp' / uuid_with_hyphens
        else:
            downloads_dir = Path(__file__).parent / 'tmp'
        downloads_dir.mkdir(parents=True, exist_ok=True)
        output_file = downloads_dir / filename
    
    # Download file
    print(f"Downloading from S3: s3://{bucket}/{s3_key}")
    s3_client.download_file(bucket, s3_key, str(output_file))
    
    # Get file size
    file_size = output_file.stat().st_size
    print(f"✓ Downloaded to: {output_file}")
    print(f"  File size: {file_size:,} bytes")
    
    return output_file


async def download_multiple_from_s3(files: list, output_dir: Optional[Path] = None, uuid_with_hyphens: Optional[str] = None) -> list:
    """
    Download multiple files from S3.
    
    Args:
        files: List of tuples [(s3_key, content_type), ...]
        output_dir: Optional output directory
        uuid_with_hyphens: UUID with hyphens for directory naming
    
    Returns:
        List of downloaded file paths
    """
    if output_dir:
        downloads_dir = output_dir
    else:
        # Download to /tmp/{uuid-with-hyphens}/
        if uuid_with_hyphens:
            downloads_dir = Path(__file__).parent / 'tmp' / uuid_with_hyphens
        else:
            downloads_dir = Path(__file__).parent / 'tmp'
    
    downloads_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    print(f"\nDownloading {len(files)} file(s) from S3...")
    
    for i, (s3_key, content_type) in enumerate(files, 1):
        filename = Path(s3_key).name
        output_file = downloads_dir / filename
        
        print(f"\n[{i}/{len(files)}] Downloading: {filename}")
        downloaded_path = await download_from_s3(s3_key, output_file, uuid_with_hyphens)
        downloaded_files.append(downloaded_path)
    
    return downloaded_files


async def download_from_ib(asset_id: str, output_path: Optional[Path] = None) -> Path:
    """
    Download file from Intelligence Bank.
    
    Args:
        asset_id: Asset UUID
        output_path: Optional output path
    
    Returns:
        Path to downloaded file
    """
    print(f"Downloading from Intelligence Bank...")
    
    # Import here to avoid circular dependency
    from intelligencebank_utils.download_asset import download_asset
    
    output_str = str(output_path) if output_path else None
    return await download_asset(asset_id, output_str)


async def smart_download(asset_id: str, output_path: Optional[str] = None, force_ib: bool = False, return_json: bool = False):
    """
    Smart download: checks S3 first, falls back to Intelligence Bank.
    
    Args:
        asset_id: Asset UUID (with or without dashes)
        output_path: Optional output path
        force_ib: Force download from IB even if S3 has the file
        return_json: Return detailed JSON metadata instead of just file path
    
    Returns:
        Path to downloaded file, or dict with metadata if return_json=True
    """
    print(f"Smart download for asset: {asset_id}")
    
    # Normalize UUID and add hyphens for directory naming
    normalized_id = normalize_uuid(asset_id)
    uuid_with_hyphens = add_uuid_hyphens(normalized_id)
    
    output_file = Path(output_path) if output_path else None
    
    # Initialize result metadata
    result = {
        'asset_id': asset_id,
        'uuid': normalized_id,
        'uuid_with_hyphens': uuid_with_hyphens,
        'source': None,  # 's3' or 'ib'
        'files': [],
        'download_dir': None,
        'asset_type': None,
        'mime_type': None
    }
    
    if not force_ib:
        # First, try to get resource metadata to determine asset type
        try:
            resource = await get_resource_by_uuid(normalized_id)
            
            if resource:
                # Determine asset type from file MIME type
                file_info = resource.get('file', {})
                mime_type = file_info.get('type', '')
                asset_type = get_asset_type_from_mime(mime_type)
                result['mime_type'] = mime_type
                result['asset_type'] = asset_type
                print(f"Asset type: {asset_type} (MIME: {mime_type})")
                
                # Check S3 with specific asset type and MIME filtering
                s3_result = await check_s3_for_asset(asset_id, mime_type, asset_type)
            else:
                # Try all asset types without MIME filtering
                s3_result = await check_s3_for_asset(asset_id)
        except:
            # If IB check fails, still try S3
            s3_result = await check_s3_for_asset(asset_id)
        
        if s3_result:
            result['source'] = 's3'
            
            # s3_result is now a list of (s3_key, content_type) tuples
            if len(s3_result) == 1:
                # Single file - download to /tmp/{uuid-with-hyphens}/
                s3_key, content_type = s3_result[0]
                downloaded_path = await download_from_s3(s3_key, output_file, uuid_with_hyphens)
                
                result['download_dir'] = str(downloaded_path.parent)
                result['files'].append({
                    'filename': downloaded_path.name,
                    'path': str(downloaded_path),
                    'size': downloaded_path.stat().st_size,
                    'content_type': content_type,
                    's3_key': s3_key
                })
                
                return result if return_json else downloaded_path
            else:
                # Multiple files - download all to /tmp/{uuid-with-hyphens}/
                output_dir = output_file.parent if output_file else None
                downloaded_files = await download_multiple_from_s3(s3_result, output_dir, uuid_with_hyphens)
                print(f"\n✓ Downloaded {len(downloaded_files)} file(s)")
                
                if downloaded_files:
                    result['download_dir'] = str(downloaded_files[0].parent)
                    
                    for i, downloaded_path in enumerate(downloaded_files):
                        s3_key, content_type = s3_result[i]
                        result['files'].append({
                            'filename': downloaded_path.name,
                            'path': str(downloaded_path),
                            'size': downloaded_path.stat().st_size,
                            'content_type': content_type,
                            's3_key': s3_key
                        })
                
                return result if return_json else (downloaded_files[0] if downloaded_files else None)
        
        print("✗ Not found in S3, falling back to Intelligence Bank...")
    
    # Fallback to Intelligence Bank
    result['source'] = 'ib'
    downloaded_path = await download_from_ib(asset_id, output_file)
    
    if downloaded_path:
        result['download_dir'] = str(downloaded_path.parent)
        result['files'].append({
            'filename': downloaded_path.name,
            'path': str(downloaded_path),
            'size': downloaded_path.stat().st_size,
            'content_type': 'unknown',
            's3_key': None
        })
    
    return result if return_json else downloaded_path


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Smart download from S3 or Intelligence Bank',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Try S3 first, fallback to IB
  python download_smart.py fca0a091-a6d0-41d5-8a26-1932a9aa6de1
  
  # Return JSON metadata
  python download_smart.py fca0a091-a6d0-41d5-8a26-1932a9aa6de1 --json
  
  # Force download from IB
  python download_smart.py fca0a091-a6d0-41d5-8a26-1932a9aa6de1 --force-ib
  
  # Custom output path
  python download_smart.py fca0a091-a6d0-41d5-8a26-1932a9aa6de1 -o myfile.pdf
        """
    )
    parser.add_argument('asset_id', help='Asset UUID (with or without dashes)')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('--force-ib', action='store_true', help='Force download from Intelligence Bank')
    parser.add_argument('--json', action='store_true', help='Return JSON metadata about downloaded files')
    
    args = parser.parse_args()
    
    try:
        result = await smart_download(args.asset_id, args.output, args.force_ib, args.json)
        
        if args.json:
            # Output JSON to stdout
            import json
            print(json.dumps(result, indent=2))
        else:
            print(f"\n✓ Success! Asset downloaded to: {result}")
        
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
