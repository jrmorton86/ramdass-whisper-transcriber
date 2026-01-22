# Multi-File Upload with Batch Processing Design

## Overview

Add multi-file upload capability with visual progress feedback, automatic MP3 conversion, and multi-GPU batch processing.

## Goals

- Upload multiple files at once (drag-drop or folder)
- Show real-time upload progress per file and overall
- Convert all audio/video formats to MP3 before transcription
- Distribute jobs across multiple GPUs efficiently
- Handle failures gracefully without stopping the batch
- Clean up local files after processing (S3 upload scaffolded for future)

## Frontend Changes

### File Selection

Replace single-file upload with multi-file drop zone:

- Accept multiple files via drag-drop or file picker (`multiple` attribute)
- Support folder drops via `webkitdirectory` API toggle
- Recursively extract audio/video files from dropped folders
- Show file queue with: filename, size, format icon, remove button

### Upload Progress UI

```
┌─────────────────────────────────────────────────┐
│  Drop files or folders here                     │
│  ─────────────────────────────────────────────  │
│  Overall: ████████░░░░░░░░ 45% (234MB / 520MB)  │
│  Files: 3 of 7 uploading                        │
│                                                 │
│  ▼ Show details                                 │
│  ┌─────────────────────────────────────────┐    │
│  │ ✓ lecture1.mp3      45MB    Complete    │    │
│  │ ✓ lecture2.wav      120MB   Complete    │    │
│  │ ↻ lecture3.m4a      89MB    45% ████░░  │    │
│  │ ○ lecture4.mp4      156MB   Pending     │    │
│  │ ○ video_folder/...  110MB   Pending     │    │
│  └─────────────────────────────────────────┘    │
│                                                 │
│  [Cancel]                      [Upload All]     │
└─────────────────────────────────────────────────┘
```

### Upload Method

Single streaming request per file using XMLHttpRequest with upload.onprogress:

```typescript
const xhr = new XMLHttpRequest();
xhr.upload.onprogress = (e) => {
  const percent = (e.loaded / e.total) * 100;
  updateFileProgress(file.id, percent);
};
xhr.open('POST', '/api/upload');
xhr.send(formData);
```

- Single HTTP request per file (no chunking)
- Backend receives file exactly as today
- Real-time progress without polling
- 2-3 concurrent uploads client-side

### Batch Summary UI

After all jobs finish:

```
┌─────────────────────────────────────────────────┐
│  Batch Complete                                 │
│  ─────────────────────────────────────────────  │
│  ✓ 5 succeeded                                  │
│  ✗ 2 failed                                     │
│                                                 │
│  Failed jobs:                                   │
│  • corrupt_file.mp3 - FFmpeg conversion failed  │
│  • too_short.wav - No speech detected           │
│                                                 │
│  [Retry Failed]  [Download All]  [Dismiss]      │
└─────────────────────────────────────────────────┘
```

## Backend Changes

### FFmpeg Conversion

After saving uploaded file, convert to MP3 using asyncio.create_subprocess_exec (safe from shell injection):

```python
async def convert_to_mp3(input_path: Path, output_path: Path) -> bool:
    """Convert any audio/video file to MP3 using ffmpeg."""
    cmd = [
        "ffmpeg", "-i", str(input_path),
        "-vn",              # Strip video
        "-acodec", "libmp3lame",
        "-ar", "16000",     # 16kHz sample rate (Whisper preferred)
        "-ac", "1",         # Mono
        "-q:a", "2",        # Quality (0-9, lower = better)
        str(output_path),
        "-y"                # Overwrite
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await process.wait()
    return process.returncode == 0
```

Flow: Upload file.wav -> Save temp -> ffmpeg -> file.mp3 -> Create job -> Delete temp .wav

### Multi-GPU Job Distribution

Least-busy-first scheduling:

```python
class JobManager:
    def __init__(self):
        self.gpu_workers: dict[int, GPUWorker] = {}
        self.gpu_busy_until: dict[int, datetime] = {}

    def get_least_busy_gpu(self) -> int:
        """Return GPU that will be free soonest."""
        now = datetime.utcnow()
        return min(
            self.gpu_busy_until.keys(),
            key=lambda gpu: self.gpu_busy_until.get(gpu, now)
        )

    async def assign_job(self, job: Job):
        gpu_id = self.get_least_busy_gpu()
        estimated_duration = self.estimate_duration(job)
        self.gpu_busy_until[gpu_id] = datetime.utcnow() + estimated_duration
        await self.gpu_workers[gpu_id].submit(job)
```

Dashboard shows GPU assignment:
```
Job: lecture1.mp3  |  Status: Processing  |  GPU: 0  |  Progress: 45%
Job: lecture2.wav  |  Status: Processing  |  GPU: 1  |  Progress: 23%
Job: lecture3.m4a  |  Status: Pending     |  GPU: -  |  Queue position: 1
```

### Batch Tracking

New batch model and endpoints:

```python
class Batch:
    id: str
    job_ids: list[str]
    created_at: datetime

# Endpoints
POST /api/batch -> returns batch_id
GET /api/batch/{batch_id} -> returns all jobs with summary
```

### S3 Upload (Scaffolded)

Placeholder for future S3 integration:

```python
async def upload_to_s3(job_id: str, output_dir: Path) -> dict:
    """Upload results to S3. Currently scaffolded."""

    files_to_upload = [
        output_dir / f"{base_name}_formatted_refined.txt",
        output_dir / f"{base_name}_refined.srt",
        output_dir / f"{base_name}.json",
    ]

    # TODO: Implement actual S3 upload
    await self._log("info", "[S3] Upload step scaffolded - skipping")

    return {"uploaded": False, "bucket": None, "keys": []}
```

### Cleanup

Delete all local files after processing:

```python
async def cleanup_job_files(job_id: str, input_path: Path, output_dir: Path):
    """Remove all local files after successful processing."""
    if input_path.exists():
        input_path.unlink()
    if output_dir.exists():
        shutil.rmtree(output_dir)
```

Cleanup triggers:
- Job completes successfully (after S3 step)
- Job fails (keep briefly for debugging, then cleanup)
- Job cancelled (immediate cleanup)

### Error Handling

- Each job runs independently
- Failures don't stop other jobs in batch
- Batch summary shows succeeded/failed counts
- "Retry Failed" re-submits only failed jobs

## Files to Modify/Create

| File | Change |
|------|--------|
| `packages/frontend/src/components/JobSubmission.tsx` | Multi-file UI with progress |
| `packages/frontend/src/lib/api.ts` | Batch endpoints, XHR upload |
| `packages/api/app/routers/files.py` | FFmpeg conversion step |
| `packages/api/app/routers/batch.py` | New batch endpoints |
| `packages/api/app/services/job_manager.py` | Multi-GPU scheduler |
| `packages/api/app/services/s3_upload.py` | Scaffolded S3 module |
| `packages/api/app/schemas/batch.py` | Batch model |

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| XHR over chunked uploads | Single request, simpler server, native progress events |
| FFmpeg to 16kHz mono MP3 | Whisper optimal format, reduces storage |
| Least-busy-first GPU | Better load balancing for varied file lengths |
| Continue on failure | Batch processing shouldn't stop for one bad file |
| Scaffold S3 | Future-proof architecture without blocking current work |
