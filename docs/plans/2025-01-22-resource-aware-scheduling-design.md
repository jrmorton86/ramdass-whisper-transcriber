# Resource-Aware Job Scheduling Design

## Goal

Enable multiple concurrent transcription jobs per GPU based on available VRAM, while monitoring system RAM to prevent instability. Target: 5 concurrent jobs (3 on RTX 5070 Ti + 2 on RTX 2080 Super) instead of current 2.

## Architecture

### Current State
- One processor task per GPU pulls from shared queue when idle
- Maximum 2 concurrent jobs (1 per GPU)
- No resource monitoring

### New State
- Single scheduler task assigns jobs to GPUs with available slots
- ResourceManager tracks slots per GPU and system RAM
- Maximum 5 concurrent jobs based on VRAM capacity

```
                    ┌─────────────┐
                    │  Scheduler  │
                    │    Task     │
                    └──────┬──────┘
                           │ assigns jobs
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │ GPU 0      │  │ GPU 0      │  │ GPU 1      │
    │ Slot 0     │  │ Slot 1     │  │ Slot 0     │
    │ (job task) │  │ (job task) │  │ (job task) │
    └────────────┘  └────────────┘  └────────────┘
```

## Resource Constraints

### GPU VRAM
| GPU | Total VRAM | Headroom | Available | Slots (3GB/job) |
|-----|------------|----------|-----------|-----------------|
| RTX 5070 Ti | 12,227 MB | 2,000 MB | 10,227 MB | 3 |
| RTX 2080 Super | 8,192 MB | 2,000 MB | 6,192 MB | 2 |

### System RAM
- Total: 32 GB
- Minimum free before starting job: 4 GB
- Historical observation: instability at 5+ concurrent jobs without RAM check

### Model Requirements
- Whisper medium model: ~3 GB VRAM per job
- System RAM per job: ~1-2 GB

## Components

### ResourceManager (`services/resource_manager.py`)

```python
class ResourceManager:
    VRAM_PER_JOB_MB = 3000      # ~3GB for medium model
    VRAM_HEADROOM_MB = 2000     # 2GB safety buffer per GPU
    MIN_SYSTEM_RAM_MB = 4000    # Require 4GB free before starting job

    async def initialize(self):
        """Detect GPUs and calculate slots at startup."""

    async def wait_for_slot(self) -> int:
        """Wait for available GPU slot, return GPU ID."""

    async def reserve_slot(self, gpu_id: int, job_id: str) -> bool:
        """Reserve a slot on GPU for job."""

    async def release_slot(self, gpu_id: int, job_id: str):
        """Release slot when job completes."""

    def check_system_ram(self) -> bool:
        """Return True if enough system RAM available."""

    async def wait_for_ram(self, timeout: float = 300):
        """Wait for RAM to free up, timeout after 5 minutes."""

    def get_status(self) -> dict:
        """Return current resource status for dashboard."""
```

### GPU Detection

Query at startup using nvidia-smi:
```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits
```

Calculate max slots per GPU:
```python
max_slots = (total_vram_mb - VRAM_HEADROOM_MB) // VRAM_PER_JOB_MB
```

### System RAM Check

Using psutil before each job:
```python
import psutil
available_mb = psutil.virtual_memory().available / (1024 * 1024)
return available_mb >= MIN_SYSTEM_RAM_MB
```

## JobManager Changes

Replace per-GPU processor tasks with single scheduler:

```python
async def _scheduler_loop(self):
    while self._running:
        job = await self.pending_jobs.get()

        # Wait for available GPU slot
        gpu_id = await self._resource_manager.wait_for_slot()

        # Wait for system RAM if needed
        if not self._resource_manager.check_system_ram():
            await self._resource_manager.wait_for_ram()

        # Reserve and spawn job task
        await self._resource_manager.reserve_slot(gpu_id, job["id"])
        task = asyncio.create_task(self._run_job_with_cleanup(job, gpu_id))

async def _run_job_with_cleanup(self, job: dict, gpu_id: int):
    try:
        await self._run_job(job, gpu_id)
    finally:
        await self._resource_manager.release_slot(gpu_id, job["id"])
```

## Error Handling

### GPU Detection Failure
Fall back to single GPU with 1 slot if nvidia-smi fails.

### Job Crash
`finally` block ensures slot is always released.

### RAM Timeout
Wait up to 5 minutes for RAM, then proceed anyway with warning.

### Cancellation
Cancelled jobs release slots via `finally` block.

## API Endpoint

`GET /api/resources`

```json
{
  "gpus": [
    {"id": 0, "name": "RTX 5070 Ti", "total_mb": 12227, "slots_used": 2, "slots_max": 3},
    {"id": 1, "name": "RTX 2080 Super", "total_mb": 8192, "slots_used": 1, "slots_max": 2}
  ],
  "system_ram_mb": 32768,
  "system_ram_available_mb": 24500,
  "pending_jobs": 4
}
```

## Logging

### Startup
```
INFO: GPU 0 (RTX 5070 Ti): 12227 MB VRAM, 3 slots available
INFO: GPU 1 (RTX 2080 Super): 8192 MB VRAM, 2 slots available
INFO: Resource manager initialized: 5 total slots across 2 GPUs
```

### Per-Job
```
INFO: Job abc123 assigned to GPU 1 (slot 2/2 now used)
INFO: Job abc123 completed, GPU 1 slot released (1/2 now used)
```

### Contention
```
INFO: All GPU slots busy, job def456 queued
WARN: System RAM low (3.2 GB free), waiting before starting job...
```

## Files to Modify

| File | Action |
|------|--------|
| `services/resource_manager.py` | Create |
| `services/job_manager.py` | Modify |
| `routers/resources.py` | Create |
| `main.py` | Modify |
| `requirements.txt` | Add psutil |

## Not Included (YAGNI)

- Live VRAM queries (track our own jobs instead)
- Model size switching (only using medium)
- CPU core tracking (not the bottleneck)
- Per-job RAM estimation (use fixed threshold)
