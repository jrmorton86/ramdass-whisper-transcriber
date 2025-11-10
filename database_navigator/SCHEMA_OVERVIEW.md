# Database Schema Overview & Next Steps

## Connection Summary
✅ **Successfully connected to RDS PostgreSQL database!**

- **Database**: `dam_ramdass_io_rds` (PostgreSQL 17.4)
- **Connection Method**: SSH Tunnel through bastion host
- **Tunnel Command**: 
  ```bash
  ssh -i ~/.ssh/ramdass-bastion-temp.pem -N -L 5433:dam-ramdass-io-rds-instance-1.c7ecmfdohgux.us-east-1.rds.amazonaws.com:5432 ec2-user@54.175.205.16
  ```
- **Local Connection**: `localhost:5433`
- **Bastion IP**: `54.175.205.16` (Elastic IP - persists across reboots)

## Database Stats

- **32 Tables**
- **21 Views** 
- **14 Sequences**
- **Total Rows**: ~130,000+ across all tables

## Key Tables & Row Counts

| Table | Rows | Purpose |
|-------|------|---------|
| `assets` | 22,975 | Main asset records (media files) |
| `comprehend_entities` | 51,969 | AWS Comprehend named entity recognition |
| `comprehend_key_phrases` | 40,552 | AWS Comprehend key phrase extraction |
| `asset_topics` | 10,697 | Asset-to-topic relationships |
| `asset_embeddings` | 3,648 | Vector embeddings for assets |
| `documents` | ? | Document storage (text extraction) |
| `files` | ? | File metadata |
| `teachers` | ? | Teacher/presenter records |
| `topics` | ? | Topic/category taxonomy |
| `participants` | ? | Event participants |
| `locations` | ? | Geographic locations |
| `events` | ? | Events/talks/sessions |

## Schema Highlights

### Core Content
- **Assets**: Main media asset table with rich metadata (type, format, dimensions, dates, descriptions)
- **Documents**: Extracted text content from assets
- **Files**: S3 file storage tracking

### Relationships
- **Teachers**: Asset-to-teacher mapping (Ram Dass talks/lectures)
- **Topics**: Hierarchical topic taxonomy 
- **Participants**: People featured in assets
- **Locations**: Geographic tagging
- **Events**: Event/session metadata

### AI/ML Features
- **Embeddings**: Vector embeddings using Amazon Titan (3,648 embeddings)
- **Comprehend Entities**: Named entity recognition (~52k entities)
- **Comprehend Key Phrases**: Key phrase extraction (~41k phrases)  
- **Comprehend Sentiment**: Sentiment analysis (912 records)
- **Full-Text Search**: `search_vector` tsvector column on assets

### Custom Metadata System
- Flexible custom fields and filters via `ib_*` tables
- Form-based filtering system
- Tag/taxonomy management

### Sync Infrastructure  
- `sync_logs` and `sync_metadata` for data pipeline tracking
- File history tracking
- Bynder DAM integration fields

## 21 Pre-built Views

The database includes sophisticated views for analytics:
- `v_assets_complete` - Full asset details
- `v_asset_stats_by_decade` - Temporal analysis
- `v_assets_by_teacher` - Content by teacher
- `v_assets_by_topic` - Content by topic
- `v_top_locations`, `v_top_organizations`, `v_top_people` - Entity aggregations
- `v_asset_sentiment_overview` - Sentiment analysis
- `vw_embedding_stats` - Embedding coverage

## Recommended Next Steps

### 1. Data Exploration Tools
- **Table browser** - Navigate tables, view sample data
- **Query interface** - Run SQL queries with results export
- **View explorer** - Leverage pre-built analytics views
- **Schema visualizer** - Generate ERD diagrams

### 2. Content Analysis
- **Asset discovery** - Browse/search the 23k assets
- **Transcript search** - Full-text search across documents
- **Entity browser** - Explore the 52k named entities
- **Topic explorer** - Navigate topic hierarchy
- **Teacher catalog** - Browse content by teacher

### 3. AI/ML Features
- **Semantic search** - Query using vector embeddings
- **Entity extraction** - Analyze comprehend entities
- **Sentiment dashboard** - Visualize sentiment patterns
- **Embedding coverage** - Identify assets without embeddings

### 4. Data Export/Migration
- **Export scripts** - Export data to JSON/CSV
- **Backup tools** - Full database backup
- **Data transfer** - Migrate to new system
- **Archive preparation** - Package for long-term storage

### 5. API/Integration
- **REST API** - Expose database via API
- **GraphQL endpoint** - Flexible querying
- **Search service** - Elasticsearch integration
- **Webhook system** - Event notifications

## Quick Start Commands

### Start SSH Tunnel (Required for all database access)
```powershell
# Option 1: Using PowerShell job (background)
Start-Job -ScriptBlock { 
  ssh -i "$env:USERPROFILE\.ssh\ramdass-bastion-temp.pem" `
      -o StrictHostKeyChecking=no `
      -o ServerAliveInterval=60 `
      -N -L 5433:dam-ramdass-io-rds-instance-1.c7ecmfdohgux.us-east-1.rds.amazonaws.com:5432 `
      ec2-user@54.175.205.16 
}

# Option 2: Using Python script
python -m database_navigator.ssh_tunnel

# Check tunnel is working
python -m database_navigator.test_connection
```

### Run Test Scripts
```powershell
# Test database connection
python -m database_navigator.test_connection

# Test tunnel + connection
python -m database_navigator.test_tunnel

# Re-extract schema (if changes are made)
python -m database_navigator.extract_schema
```

### Python Connection Example
```python
from database_navigator import get_connection

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM assets")
        count = cur.fetchone()[0]
        print(f"Total assets: {count}")
```

## Infrastructure Notes

### Bastion Host
- **Instance ID**: `i-0bb00f480df594905`
- **Type**: t2.micro (free tier eligible)
- **AMI**: Amazon Linux 2023
- **Elastic IP**: `54.175.205.16`
- **Security Group**: `sg-05bdffcb60fef2797`
- **Key**: `~/.ssh/ramdass-bastion-temp.pem`

### Security Group Rules
- Port 22 (SSH): From your IP (75.63.121.219/32)
- Port 5432 (PostgreSQL): 
  - From your IP (75.63.121.219/32)
  - From bastion security group (self-reference)

### RDS Instance
- **Cluster**: `dam-ramdass-io-rds`
- **Instance**: `dam-ramdass-io-rds-instance-1`
- **Engine**: Aurora PostgreSQL 17.4
- **Type**: db.serverless
- **Status**: Available, publicly accessible (through bastion only)

### Cleanup (when done)
```powershell
# Terminate bastion instance
aws ec2 terminate-instances --instance-ids i-0bb00f480df594905 --region us-east-1

# Release Elastic IP
aws ec2 release-address --allocation-id eipalloc-0556309c3979d6c39 --region us-east-1

# Delete key pair
aws ec2 delete-key-pair --key-name ramdass-bastion-temp --region us-east-1
Remove-Item "$env:USERPROFILE\.ssh\ramdass-bastion-temp.pem"
```

## Files Created

- `.env` - Database connection configuration
- `database_navigator/config.py` - Settings management with Secrets Manager
- `database_navigator/db.py` - Database connection utilities
- `database_navigator/extract_schema.py` - Schema extraction tool
- `database_navigator/test_connection.py` - Connection diagnostics
- `database_navigator/test_tunnel.py` - Tunnel testing tool
- `database_navigator/ssh_tunnel.py` - SSH tunnel helper
- `database_navigator/schema_output/schema_*.json` - Full schema (JSON)
- `database_navigator/schema_output/schema_summary_*.txt` - Readable schema summary

---

**What would you like to build next?** 🚀
