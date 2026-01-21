# Audio Asset Embeddings System

Complete system for generating embeddings and AWS Comprehend analysis for audio transcripts.

## 📋 Overview

This system processes audio transcripts to generate:
- **4 Embeddings per asset** (transcript, summary, keywords, metadata)
- **AWS Comprehend Analysis** (entities, key phrases, sentiment)

All data is stored in PostgreSQL for semantic search and filtering.

## 🏗️ Architecture

```
Audio Asset (with transcript in S3)
          ↓
fetch_transcript_from_s3()
          ↓
generate_summary_and_keywords() ← Claude via Bedrock
          ↓
generate_embedding() × 4 ← Amazon Titan
          ↓
analyze_transcript_with_comprehend() ← AWS Comprehend
          ↓
store_embeddings() + store_comprehend_analysis()
          ↓
Database (5 tables populated)
```

## 📦 Components

### Core Module: `comprehend_utils/`

```
comprehend_utils/
├── __init__.py              # Module exports
├── config.py                # AWS configuration
├── comprehend.py            # Core analysis functions
└── generate_audio_embeddings.py  # Main processing logic
```

### Scripts

- **`test_embedding_generation.py`** - Test single asset
- **`batch_process_embeddings.py`** - Process multiple assets

## 🚀 Quick Start

### 1. Process a Single Asset

```bash
# Test with one asset
python test_embedding_generation.py <asset-uuid>

# Force regenerate existing embeddings
python test_embedding_generation.py <asset-uuid> --force

# Skip Comprehend (faster, embeddings only)
python test_embedding_generation.py <asset-uuid> --skip-comprehend
```

### 2. Batch Process Multiple Assets

```bash
# Process first 10 audio assets without embeddings
python batch_process_embeddings.py --limit 10

# Process all audio assets without embeddings
python batch_process_embeddings.py

# Process with custom delay (rate limiting)
python batch_process_embeddings.py --limit 50 --delay 2.0

# Skip Comprehend for faster processing
python batch_process_embeddings.py --limit 100 --skip-comprehend
```

### 3. Process Specific Assets from File

Create a text file with UUIDs (one per line):

```text
# my_assets.txt
550e8400-e29b-41d4-a716-446655440000
fca0a091-a6d0-41d5-8a26-1932a9aa6de1
# Comments are ignored
```

Then run:

```bash
python batch_process_embeddings.py --uuids-file my_assets.txt
```

## 📊 Data Storage

### Embeddings (asset_embeddings table)

Each asset gets 4 embedding records:

| content_type | Description | Text Source |
|--------------|-------------|-------------|
| `transcript` | Full transcript | First 8000 chars of transcript |
| `summary` | Claude summary | Generated 3-5 paragraph summary |
| `keywords` | Keywords | Generated 10-15 keywords |
| `metadata` | Asset metadata | "Name. Keywords: x. Type: Audio" |

**Schema:**
```sql
asset_embeddings (
    id UUID PRIMARY KEY,
    resource_id UUID,  -- asset.id
    content_type TEXT, -- transcript/summary/keywords/metadata
    embedding VECTOR(1024),  -- Titan embedding
    embedding_model TEXT,    -- 'amazon.titan-embed-text-v2:0'
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)
```

### Comprehend Analysis (3 tables)

#### comprehend_entities
```sql
comprehend_entities (
    id UUID PRIMARY KEY,
    resource_id UUID,
    entity_text TEXT,
    entity_type TEXT,  -- PERSON, LOCATION, ORGANIZATION, etc.
    confidence_score FLOAT,
    begin_offset INT,
    end_offset INT
)
```

#### comprehend_key_phrases
```sql
comprehend_key_phrases (
    id UUID PRIMARY KEY,
    resource_id UUID,
    phrase_text TEXT,
    confidence_score FLOAT,
    begin_offset INT,
    end_offset INT
)
```

#### comprehend_sentiment
```sql
comprehend_sentiment (
    id UUID PRIMARY KEY,
    resource_id UUID,
    sentiment TEXT,  -- POSITIVE, NEGATIVE, NEUTRAL, MIXED
    positive_score FLOAT,
    negative_score FLOAT,
    neutral_score FLOAT,
    mixed_score FLOAT
)
```

## 🔧 Programmatic Usage

### Process Single Asset

```python
from comprehend_utils.generate_audio_embeddings import generate_audio_embeddings

result = generate_audio_embeddings(
    asset_uuid='550e8400-e29b-41d4-a716-446655440000',
    force=False,           # Set True to regenerate
    skip_comprehend=False  # Set True to skip Comprehend
)

print(f"Success: {result['success']}")
print(f"Embeddings: {result['embeddings_created']}")
print(f"Entities: {result['comprehend_entities']}")
print(f"Phrases: {result['comprehend_phrases']}")
```

### Use Individual Functions

```python
from comprehend_utils import (
    generate_embedding,
    analyze_transcript_with_comprehend,
    store_embeddings,
    store_comprehend_analysis
)

# Generate single embedding
text = "Ram Dass teaches about love and service..."
embedding = generate_embedding(text)  # Returns 1024-dim vector

# Analyze with Comprehend
analysis = analyze_transcript_with_comprehend(transcript_text)
# Returns: {'entities': [...], 'key_phrases': [...], 'sentiment': {...}}

# Store in database
from database_navigator import get_connection

with get_connection() as conn:
    # Store embeddings
    embeddings_data = [{
        'content_type': 'transcript',
        'embedding': embedding,
    }]
    store_embeddings(conn, asset_uuid, embeddings_data)
    
    # Store Comprehend analysis
    store_comprehend_analysis(conn, asset_uuid, analysis)
```

## 📈 Processing Statistics

### Performance

- **Transcript embedding**: ~1-2 seconds
- **Claude summary/keywords**: ~5-10 seconds
- **4 Titan embeddings**: ~4-8 seconds
- **Comprehend analysis**: ~10-30 seconds (depending on length)
- **Database storage**: ~1 second

**Total per asset**: ~20-50 seconds

### Costs (AWS)

- **Bedrock Claude Sonnet**: ~$0.003 per 1K tokens (summary/keywords)
- **Bedrock Titan Embeddings**: ~$0.0001 per 1K tokens (4 embeddings)
- **Comprehend**: ~$0.0001 per 100 chars (entities/phrases/sentiment)

**Estimated cost per audio asset**: $0.01 - $0.05

## ⚙️ Configuration

### Environment Variables

Required in `.env`:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret

# S3 Bucket
S3_BUCKET=dam-ramdass-io-assets

# Database (managed by database_navigator)
DB_HOST=your-rds-host
DB_PORT=5432
DB_NAME=dam_ramdass_io_rds
# ... other DB settings
```

### AWS Permissions Required

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "comprehend:DetectEntities",
        "comprehend:DetectKeyPhrases",
        "comprehend:DetectSentiment",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": "*"
    }
  ]
}
```

## 🔍 Monitoring & Debugging

### Check Embedding Coverage

```python
from database_navigator import get_connection

with get_connection() as conn:
    with conn.cursor() as cur:
        # Count audio assets without embeddings
        cur.execute("""
            SELECT COUNT(*)
            FROM assets
            WHERE asset_type = 'Audio'
            AND id NOT IN (
                SELECT DISTINCT resource_id 
                FROM asset_embeddings
            )
        """)
        count = cur.fetchone()[0]
        print(f"Audio assets needing embeddings: {count}")
```

### View Processing Results

```bash
# Batch processing creates timestamped results files
ls batch_embeddings_results_*.json

# View results
cat batch_embeddings_results_20251110_143022.json
```

### Common Issues

**Issue**: `Asset not found or is not Audio type`
- **Solution**: Verify UUID and check `asset_type = 'Audio'` in database

**Issue**: `No transcript found in S3`
- **Solution**: Ensure transcript was created first via transcription pipeline

**Issue**: `Embeddings already exist`
- **Solution**: Use `--force` flag to regenerate

**Issue**: AWS credential errors
- **Solution**: Check `.env` file and AWS permissions

## 🚦 Future Enhancements (Async-Ready)

The system is designed for easy async expansion:

```python
# Future async mode (prepared architecture)
async def generate_audio_embeddings_async(asset_uuid: str):
    """Async version for concurrent processing."""
    # Current sync code can be wrapped with asyncio
    pass

# Batch processing with concurrency
import asyncio

async def batch_process_async(asset_uuids: List[str], max_concurrent=5):
    """Process multiple assets concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_limit(uuid):
        async with semaphore:
            return await generate_audio_embeddings_async(uuid)
    
    tasks = [process_with_limit(uuid) for uuid in asset_uuids]
    return await asyncio.gather(*tasks)
```

## 📚 Related Documentation

- **Transcription Pipeline**: `transcribe_pipeline/PIPELINE.md`
- **Database Schema**: `database_navigator/SCHEMA_OVERVIEW.md`
- **AWS Comprehend**: https://docs.aws.amazon.com/comprehend/
- **Amazon Titan Embeddings**: https://docs.aws.amazon.com/bedrock/

## 🤝 Contributing

When adding new features:

1. Update this README
2. Add tests to `test_embedding_generation.py`
3. Ensure backwards compatibility
4. Document environment variables
5. Update cost estimates

---

**Questions?** Check the code docstrings or search existing transcripts for patterns.
