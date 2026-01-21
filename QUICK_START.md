# Batch Audio Processing - Quick Start Guide

## Prerequisites

1. **SSH Key**: Ensure you have `~/.ssh/ramdass-bastion-temp.pem`
2. **Venv Activated**: Use the Python venv at `./venv`
3. **AWS Credentials**: Configured in your environment

## Recommended: Using the Batch File (Easiest)

### Step 1: Start SSH Tunnel

Open a **new terminal** and run the SSH tunnel to connect to the database:

```bash
cd c:\Users\jrmor\OneDrive\Documents\Programming\ramdass.io\transcriber
python database_navigator/ssh_tunnel.py
```

Or manually:
```bash
ssh -i ~/.ssh/ramdass-bastion-temp.pem -N -L 5433:dam-ramdass-io-rds-instance-1.c7ecmfdohgux.us-east-1.rds.amazonaws.com:5432 ec2-user@54.175.205.16
```

**Keep this terminal open while processing** - it maintains the database connection.

### Step 2: Run the Batch File

In another terminal (in the transcriber directory):

```bash
run_batch_from_json.bat
```

This will:
1. Query the database for all Audio assets without embeddings
2. Export them to `assets_without_embeddings.json`
3. Process assets with **up to 5 parallel threads** (oldest to newest)
4. Skip assets in "Resources/To Be Sorted" folder
5. **Automatically remove assets from JSON before processing** (prevents duplicates)

The batch file handles everything automatically and auto-continues on errors. Assets are removed from the JSON file immediately before processing starts, so if you stop and restart, it will continue where you left off.

---

## Alternative Method: Manual Batch Processing

### Step 1: Start SSH Tunnel (same as above)

### Step 2: Generate the JSON File

```bash
echo "2" | venv/Scripts/python.exe database_navigator/get_assets_without_embeddings.py
```

This creates `assets_without_embeddings.json` in the transcriber folder.

### Step 3: Process from JSON

```bash
venv/Scripts/python.exe batch_process_from_json.py -y -t 5 -d cuda:0
```

Options:
- `-y`: Auto-continue on errors without prompting
- `-t N`: Number of parallel threads (default: 5, adjust based on your system)
- `-d DEVICE`: CUDA device to use (e.g., `cuda:0`, `cuda:1` for multi-GPU systems)

**🔬 Experimental: Intelligent GPU Load Balancing**

For dual-GPU systems, enable automatic load balancing that distributes work based on real-time GPU utilization:

```bash
venv/Scripts/python.exe batch_process_from_json.py -y -t 10 --experimental
```

This mode:
- Monitors both `cuda:0` and `cuda:1` in real-time
- Assigns each task to the GPU with lower load
- Balances based on GPU utilization % and active task count
- Supports up to 5 tasks per GPU (adjustable with `--max-per-gpu`)

**Recommended for dual-GPU:**
```bash
# Process 10 assets concurrently across 2 GPUs (5 per GPU max)
venv/Scripts/python.exe batch_process_from_json.py -y -t 10 --experimental --max-per-gpu 5
```

---

## Alternative Method: Direct Batch Processing with Limits

If you want more control over which assets to process:

```bash
./venv/Scripts/python.exe batch_process_audio.py --limit 10 --model base
```

### Command Options

- `--limit N`: Process first N assets (e.g., `--limit 1` for testing)
- `--skip N`: Skip first N assets (default: 0)
- `--model`: Whisper model to use
  - `tiny` - fastest, lowest quality
  - `base` - recommended for most use cases
  - `small`, `medium`, `large` - slower, better quality

### Example Usage

```bash
# Test with one asset
./venv/Scripts/python.exe batch_process_audio.py --limit 1 --model base

# Process 10 assets starting after first 50
./venv/Scripts/python.exe batch_process_audio.py --limit 10 --skip 50 --model base

# Full production run (will take many hours)
./venv/Scripts/python.exe batch_process_audio.py --model base
```

## Understanding the JSON Workflow

### Why Use `batch_process_from_json.py`?

The JSON-based workflow is more efficient for processing large batches:

1. **Single Database Query**: Instead of querying the database for each asset, it generates one JSON file with all assets
2. **Resumable**: If the process stops, you can manually edit the JSON to remove processed assets and resume
3. **Filtering**: Automatically excludes "Resources/To Be Sorted" folder
4. **Reverse Order**: Processes oldest assets first (bottom to top)

### The JSON File Structure

`assets_without_embeddings.json` contains:
```json
{
  "count": 1234,
  "exported_at": "2025-11-11T12:00:00",
  "filter": {"asset_type": "Audio"},
  "assets": [
    {
      "id": "uuid-here",
      "name": "Asset Name",
      "asset_type": "Audio",
      "folder_path": "Ram Dass/1970s",
      ...
    }
  ]
}
```

## Full Pipeline Steps

Each asset goes through the following 7 steps:

1. **Download** - Fetch audio from Intelligence Bank or S3
2. **Whisper Transcription** - Convert audio to text (local model)
3. **SRT/TXT Formatting** - Generate subtitle and text files
4. **Claude Refinement** - Improve transcript formatting
5. **Generate Embeddings** - Create 4 embeddings:
   - Transcript embedding
   - Summary embedding
   - Keywords embedding
   - Metadata embedding
6. **AWS Comprehend Analysis** - Extract entities, key phrases, sentiment
7. **Upload & Store** - Save all files to S3 and database

## Output

Results are saved to `batch_audio_results_YYYYMMDD_HHMMSS.json` with:
- Total assets processed
- Successful count
- Failed count with error details
- Detailed per-asset status

## Example Session

```
Terminal 1 (SSH Tunnel):
$ python database_navigator/ssh_tunnel.py
Creating SSH tunnel to RDS...
  Bastion: ec2-user@54.175.205.16
  RDS: dam-ramdass-io-rds-instance-1.c7ecmfdohgux.us-east-1.rds.amazonaws.com:5432
  Local port: 5433
Press Ctrl+C to close the tunnel

Terminal 2 (Batch Processor):
$ ./venv/Scripts/python.exe batch_process_audio.py --limit 1 --model base
2025-11-10 04:01:36,245 - __main__ - INFO - BATCH AUDIO ASSET PROCESSOR
2025-11-10 04:01:36,245 - __main__ - INFO - Found 1 assets to process
...
```

## Troubleshooting

### "Cannot connect to database"
- Ensure SSH tunnel is running in another terminal
- Check SSH key at: `~/.ssh/ramdass-bastion-temp.pem`
- Verify `.env` has `DATABASE_HOST=localhost` and `DATABASE_PORT=5433`

### "Transcription failed"
- Check if Whisper model is downloaded for your chosen size
- Verify audio file is valid
- Check system has enough disk space for temp files

### "AWS API errors"
- Ensure AWS credentials are configured
- Check S3 bucket exists: `dam-ramdass-io-assets`
- Verify AWS region is `us-east-1`

### "Asset not found in database"
- UUID might be incorrect
- Asset might not be in the database
- Check `assets` table with: `SELECT id, name FROM assets LIMIT 5;`

## Monitoring Large Batches

For processing 22K+ assets:

```bash
# Check progress in another terminal while running
tail batch_audio_results_*.json

# Or query database to see embeddings added:
# SELECT COUNT(*) FROM asset_embeddings;
```

## Costs

Processing large batches will incur AWS costs:
- **Titan Embeddings v2**: ~$0.02 per 1M input tokens
- **AWS Comprehend**: ~$0.0001 per unit (100 chars)
- **S3**: Upload costs minimal, but storage depends on file sizes

For 22K audio assets:
- Estimate: 2-3 weeks processing time (base model, single threaded)
- Cost: $50-$200 depending on transcript lengths

Consider:
- Running with `--limit` to test first
- Monitoring AWS console for costs
- Processing in batches with `--skip`
