# Worker Pool Batch Processor Design

**Date:** 2026-01-21
**Goal:** Eliminate model loading overhead in batch processing by using persistent GPU workers

## Problem

Current batch processing spawns a new subprocess for each asset, which loads the Whisper model (~1.5GB for medium) from scratch every time. For 100 assets, this means 100 model loads instead of 2.

**Current flow:**
```
batch_process_from_json.py (ThreadPoolExecutor)
  → subprocess: batch_process_audio.py
    → subprocess: transcribe_pipeline.py
      → subprocess: whisper_with_vocab.py  ← MODEL LOADS HERE (every time!)
```

## Solution

Replace subprocess-per-asset architecture with persistent worker processes that load the model once and process many assets.

## Architecture

```
batch_process_pool.py (Main Process)
  └─ WorkerPool (2 persistent workers)
       ├─ Worker 0 (cuda:0): loads model ONCE → processes N assets
       └─ Worker 1 (cuda:1): loads model ONCE → processes N assets
  └─ TaskQueue: distributes assets to available workers
  └─ ResultCollector: gathers results, updates JSON
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  batch_process_pool.py (Main Process)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐    │
│  │ Asset Queue │→ │ WorkerPool  │→ │ Result Collector │    │
│  │ (from JSON) │  │ (2 workers) │  │ (updates JSON)   │    │
│  └─────────────┘  └─────────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────────────┘
        │                   │
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│ Worker 0      │   │ Worker 1      │
│ cuda:0        │   │ cuda:1        │
│ ┌───────────┐ │   │ ┌───────────┐ │
│ │ Whisper   │ │   │ │ Whisper   │ │
│ │ (loaded)  │ │   │ │ (loaded)  │ │
│ └───────────┘ │   │ └───────────┘ │
│ process_asset │   │ process_asset │
└───────────────┘   └───────────────┘
```

## Worker Design

```python
class TranscriptionWorker:
    def __init__(self, gpu_id, model_name="medium"):
        # Called ONCE at startup
        self.device = f"cuda:{gpu_id}"
        self.model = whisper.load_model(model_name, device=self.device)

    def process_asset(self, asset_uuid):
        # Called for EACH asset - model already loaded
        # 1. Download audio (S3/IB)
        # 2. Transcribe with self.model (no reload!)
        # 3. Format (JSON → TXT + SRT)
        # 4. Claude refinement (API call)
        # 5. Post-process (embeddings, Comprehend, DB, S3)
        # 6. Cleanup temp files
        return result
```

### What Moves Into Workers (No More Subprocesses)

| Current (subprocess) | New (direct call) |
|---------------------|-------------------|
| `whisper_with_vocab.py` | `self.model.transcribe()` inline |
| `convert_aws_transcribe.py` | Import and call `format_transcript()` |
| `claude_refine_transcript.py` | Import and call `refine_with_claude()` |
| `apply_txt_corrections_to_srt.py` | Import and call `apply_corrections()` |

## Queue Management

```
task_queue:    [uuid1, uuid2, uuid3, ..., STOP, STOP]
                  ↑                         ↑
            assets to process         sentinel per worker
```

**Natural load balancing:** Faster GPU finishes task → immediately pulls next.

### Graceful Shutdown

| Scenario | Behavior |
|----------|----------|
| Queue empty | Workers receive `STOP` sentinel, exit cleanly |
| Ctrl+C | Catch `KeyboardInterrupt`, send `STOP` to workers, wait for current tasks |
| Worker crash | Main process detects, logs error, continues with remaining workers |
| Asset failure | Worker logs error, continues to next asset |

## CLI Interface

```bash
python batch_process_pool.py [options]
```

| Flag | Description | Default |
|------|-------------|---------|
| `-t, --threads` | Number of workers (GPUs to use) | 2 |
| `-m, --model` | Whisper model size | medium |
| `-y, --yes` | Auto-continue on errors | false |
| `--skip N` | Skip first N assets | 0 |
| `--gpus` | GPU IDs to use | 0,1 |

### Example Usage

```bash
# Standard dual-GPU run
python batch_process_pool.py -t 2 --gpus 0,1

# Single GPU, large model
python batch_process_pool.py -t 1 --gpus 0 -m large

# Skip first 50, auto-continue on errors
python batch_process_pool.py --skip 50 -y
```

## Files to Create

| File | Lines (est.) | Purpose |
|------|--------------|---------|
| `worker_pool.py` | ~200 | `TranscriptionWorker` class + `WorkerPool` manager |
| `batch_process_pool.py` | ~150 | CLI entry point, queue setup, result collection |

## Files to Modify

| File | Change |
|------|--------|
| `transcribe_pipeline/whisper_with_vocab.py` | Extract `transcribe()` as importable function |
| `transcribe_pipeline/claude_refine_transcript.py` | Extract `refine()` as importable function |
| `transcribe_pipeline/convert_aws_transcribe.py` | Extract `format_transcript()` as importable function |
| `transcribe_pipeline/apply_txt_corrections_to_srt.py` | Extract `apply_corrections()` as importable function |

## Implementation Order

1. Refactor existing scripts to expose importable functions (non-breaking)
2. Create `worker_pool.py` with worker class
3. Create `batch_process_pool.py` entry point
4. Test with single worker, then dual
5. Validate against existing output quality

## What Stays the Same

- Output file formats (JSON, SRT, TXT)
- Database schema and S3 paths
- Claude refinement logic
- Post-processing (embeddings, Comprehend)
- `assets_without_embeddings.json` workflow
- Existing scripts preserved for backwards compatibility

## Expected Results

- Model loads: 2 (once per GPU) vs. N (once per asset)
- Subprocess spawns: 0 vs. ~4N
- Estimated speedup: 3-10x depending on asset count and audio length

## Hardware Requirements

- Dual GPU setup (both 8GB+ VRAM)
- Both GPUs can run Whisper medium model
- Workers treated as equivalent capacity
