# Ramdass.io Transcriber Suite

A comprehensive suite of tools for managing, downloading, and transcribing audio assets from Intelligence Bank and S3, with database navigation capabilities.

## Project Structure

```
transcriber/
├── intelligencebank_utils/    # Intelligence Bank API integration
│   ├── config.py             # IB configuration
│   ├── ib_client.py          # IB API client
│   ├── ib_discovery.py       # Asset discovery service
│   └── download_asset.py     # Direct IB download
├── s3_utils/                 # AWS S3 utilities
│   ├── config.py             # S3 configuration
│   └── s3_client.py          # S3 client operations
├── database_navigator/       # PostgreSQL database tools
│   ├── config.py             # Database configuration
│   ├── db.py                 # Database connection
│   ├── ssh_tunnel.py         # SSH tunnel for remote DB
│   ├── extract_schema.py     # Schema extraction
│   └── view_assets.py        # Asset querying
├── transcribe_pipeline/      # Audio transcription pipeline
│   └── [see PIPELINE.md]
├── download_smart.py         # Smart download (S3 → IB fallback)
├── test_ib_connection.py     # Test IB connectivity
└── config.py                 # Root configuration
```

## Features

### 🎯 Smart Asset Download
- **Automatic fallback**: Checks S3 first, falls back to Intelligence Bank
- **MIME-type filtering**: Downloads matching file types and companion files
- **Organized storage**: Files saved to `/tmp/{uuid}/`
- **JSON output**: Detailed metadata for programmatic use

### 🔌 Intelligence Bank Integration
- Session-based authentication
- Asset discovery and metadata retrieval
- Direct asset download from CDN
- UUID normalization (with/without dashes)

### ☁️ AWS S3 Integration
- Smart file detection with MIME type matching
- Extension-based fallback for generic MIME types
- Companion file detection (.json, .srt, .txt, .vtt)
- Streaming upload/download

### 🗄️ Database Navigation
- SSH tunnel support for remote databases
- Schema extraction and documentation
- Asset querying without embeddings
- PostgreSQL connection management

## Quick Start

### Prerequisites

```bash
# Python 3.12+
python --version

# Required environment variables in .env:
IB_PLATFORM_URL=ramdass.intelligencebank.com
IB_API_EMAIL=your-email@example.com
IB_API_PASSWORD=your-password
IB_SESSION_ID=your-session-id
IB_CLIENT_ID=your-client-id
IB_API_V2_URL=https://apius.intelligencebank.com
IB_API_V3_URL=https://usprod2usv3.intelligencebank.com

AWS_REGION=us-east-1
S3_BUCKET=dam-ramdass-io-assets

# Database (optional)
DB_HOST=your-db-host
DB_PORT=5432
DB_NAME=your-db-name
DB_USER=your-db-user
DB_PASSWORD=your-db-password
```

### Installation

```bash
# Clone repository
git clone <repository-url>
cd transcriber

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r transcribe_pipeline/requirements.txt
pip install boto3 psycopg2-binary httpx python-dotenv
```

## Usage

### Smart Download

Download assets with automatic S3/IB fallback:

```bash
# Basic download (tries S3 first, falls back to IB)
python download_smart.py fca0a091-a6d0-41d5-8a26-1932a9aa6de1

# Return JSON metadata
python download_smart.py fca0a091-a6d0-41d5-8a26-1932a9aa6de1 --json

# Force Intelligence Bank download
python download_smart.py fca0a091-a6d0-41d5-8a26-1932a9aa6de1 --force-ib

# Custom output path
python download_smart.py fca0a091-a6d0-41d5-8a26-1932a9aa6de1 -o myfile.pdf
```

**JSON Output Example:**
```json
{
  "asset_id": "0001d304-11dd-4246-b9bc-27199c4d6009",
  "uuid": "0001d30411dd4246b9bc27199c4d6009",
  "uuid_with_hyphens": "0001d304-11dd-4246-b9bc-27199c4d6009",
  "source": "s3",
  "files": [
    {
      "filename": "audio.wav",
      "path": "/path/to/tmp/uuid/audio.wav",
      "size": 476042770,
      "content_type": "binary/octet-stream",
      "s3_key": "audio/uuid/audio.wav"
    }
  ],
  "download_dir": "/path/to/tmp/uuid",
  "asset_type": "audio",
  "mime_type": "audio/x-wav"
}
```

### Intelligence Bank Operations

```bash
# Test connection
python test_ib_connection.py

# Direct download from IB
python -m intelligencebank_utils.download_asset fca0a091-a6d0-41d5-8a26-1932a9aa6de1
```

### Database Operations

```bash
# Test database connection
python -m database_navigator.test_connection

# View assets without embeddings
python -m database_navigator.view_assets

# Extract database schema
python -m database_navigator.extract_schema
```

## How It Works

### Smart Download Flow

1. **Normalize UUID**: Convert input to standard format (with/without dashes)
2. **Get Metadata**: Query Intelligence Bank for asset metadata
3. **Determine Type**: Extract MIME type and asset category
4. **Check S3**: 
   - Search S3 path: `s3://bucket/{type}/{uuid-with-hyphens}/`
   - Filter by MIME category (audio, video, image, documents)
   - Use extension fallback for generic MIME types
   - Include companion files (.json, .srt, .txt)
5. **Download**: 
   - If found in S3: Download all matching files
   - If not found: Fall back to Intelligence Bank CDN
6. **Organize**: Save to `/tmp/{uuid-with-hyphens}/`
7. **Return**: File path or JSON metadata

### File Type Detection

**Primary Files** (matched by MIME type):
- Audio: `.mp3`, `.wav`, `.flac`, `.aac`, `.ogg`, `.m4a`
- Video: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
- Image: `.jpg`, `.png`, `.gif`, `.bmp`, `.svg`, `.webp`
- Documents: `.pdf`, `.doc`, `.docx`, `.xls`, `.xlsx`

**Companion Files** (always included):
- `.json` - Metadata
- `.srt`, `.vtt` - Subtitles
- `.txt` - Transcripts
- `.xml` - Structured data

## API Reference

### `smart_download(asset_id, output_path=None, force_ib=False, return_json=False)`

Smart download with S3/IB fallback.

**Parameters:**
- `asset_id` (str): UUID with or without dashes
- `output_path` (str, optional): Custom output path
- `force_ib` (bool): Skip S3 check, download from IB
- `return_json` (bool): Return metadata dict instead of path

**Returns:**
- `Path` or `dict`: File path or metadata dictionary

### Intelligence Bank Client

```python
from intelligencebank_utils.ib_client import ib_client

# Authenticate
await ib_client.ensure_authenticated()

# Get headers
headers = await ib_client.get_authenticated_headers()

# Get V3 API base URL
base_url = ib_client.get_v3_base_url()
```

### S3 Client

```python
from s3_utils.s3_client import s3_client, upload_stream, download_stream

# Upload file
upload_stream(bucket, key, stream, content_type='audio/wav')

# Download file
data = download_stream(bucket, key)
```

## Configuration

### Intelligence Bank

The IB client uses:
- **Session ID**: Required for authentication (super essential)
- **Client ID**: Required for V3 API URLs
- **API V2/V3 URLs**: Base URLs for API endpoints
- Platform URL, email, password for re-authentication if needed

### AWS S3

Default configuration:
- **Bucket**: `dam-ramdass-io-assets`
- **Region**: `us-east-1`
- **Path structure**: `{type}/{uuid-with-hyphens}/filename.ext`

### Database

PostgreSQL connection via:
- Direct connection (local/accessible host)
- SSH tunnel (remote host)

## Transcription Pipeline

See [transcribe_pipeline/PIPELINE.md](transcribe_pipeline/PIPELINE.md) for detailed documentation on:
- Whisper-based transcription
- Speaker diarization
- Claude refinement
- Vocabulary management

## Development

### Running Tests

```bash
# Test IB connection
python test_ib_connection.py

# Test database connection
python -m database_navigator.test_connection

# Test SSH tunnel
python -m database_navigator.test_tunnel
```

### Adding New Features

1. Update relevant module (`intelligencebank_utils/`, `s3_utils/`, etc.)
2. Update configuration if needed
3. Add tests
4. Update documentation

## Troubleshooting

### Common Issues

**IB Connection Failed:**
- Check `IB_SESSION_ID` is valid (expires periodically)
- Verify `IB_PLATFORM_URL` and credentials
- Run `python test_ib_connection.py` to test

**S3 Access Denied:**
- Verify AWS credentials: `aws sso login`
- Check S3 bucket permissions
- Ensure `AWS_REGION` matches bucket region

**Database Connection Failed:**
- Check SSH tunnel if remote: `python -m database_navigator.test_tunnel`
- Verify database credentials
- Check firewall/security groups

**Download Not Found:**
- Asset may not exist in S3 yet
- UUID format should be valid (32 hex chars)
- Check IB has the asset

## License

See [transcribe_pipeline/LICENSE](transcribe_pipeline/LICENSE)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues, questions, or contributions, please contact the development team.
