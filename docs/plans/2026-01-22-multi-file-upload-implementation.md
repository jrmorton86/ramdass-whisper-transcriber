# Multi-File Upload Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add multi-file upload with progress tracking, FFmpeg conversion, multi-GPU distribution, and batch management.

**Architecture:** Frontend handles file selection/progress tracking via XHR, backend adds FFmpeg conversion step before job creation, job manager assigns jobs to least-busy GPU, batch model groups related jobs for summary view.

**Tech Stack:** React/TypeScript (frontend), FastAPI/Python (backend), FFmpeg (conversion), SQLAlchemy (batch model)

---

## Task 1: Backend - FFmpeg Conversion Utility

**Files:**
- Create: `packages/api/app/services/ffmpeg.py`

**Step 1: Create FFmpeg conversion module**

Create a module that uses `asyncio.create_subprocess_exec` (safe from shell injection) to convert audio/video files to MP3 format optimized for Whisper (16kHz mono).

Key functions:
- `convert_to_mp3(input_path, output_path)` - Returns tuple of (success, error_message)
- `get_audio_duration(file_path)` - Returns duration in seconds for job estimation

**Step 2: Commit**

```bash
git add packages/api/app/services/ffmpeg.py
git commit -m "feat: add FFmpeg conversion utility"
```

---

## Task 2: Backend - Integrate FFmpeg into Upload Endpoint

**Files:**
- Modify: `packages/api/app/routers/files.py`

**Step 1: Update upload endpoint with conversion**

Modify the upload flow:
1. Save uploaded file to temp location with original extension
2. If not already MP3, convert using FFmpeg
3. Delete temp file, keep converted MP3
4. Create job pointing to final MP3

Add optional `batch_id` query parameter for grouping jobs.

**Step 2: Commit**

```bash
git add packages/api/app/routers/files.py
git commit -m "feat: integrate FFmpeg conversion into upload endpoint"
```

---

## Task 3: Backend - Batch Model and Schema

**Files:**
- Create: `packages/api/app/schemas/batch.py`
- Modify: `packages/api/app/schemas/job.py` (add batch_id field)

**Step 1: Create batch schema**

Batch model with:
- id (UUID primary key)
- name (optional display name)
- created_at (timestamp)

**Step 2: Add batch_id to Job schema**

Add `batch_id = Column(String(36), nullable=True)` to Job model.
Update `to_dict()` to include batchId.

**Step 3: Commit**

```bash
git add packages/api/app/schemas/batch.py packages/api/app/schemas/job.py
git commit -m "feat: add Batch model and link jobs to batches"
```

---

## Task 4: Backend - Batch API Endpoints

**Files:**
- Create: `packages/api/app/routers/batch.py`
- Create: `packages/api/app/models/batch.py`
- Modify: `packages/api/app/main.py` (register router)

**Step 1: Create batch Pydantic models**

- BatchCreate (request)
- BatchResponse (single batch)
- BatchSummary (with job counts by status)

**Step 2: Create batch router**

Endpoints:
- POST /api/batch - Create new batch
- GET /api/batch/{batch_id} - Get batch with summary
- GET /api/batch/{batch_id}/jobs - Get all jobs in batch
- POST /api/batch/{batch_id}/retry-failed - Retry failed jobs

**Step 3: Register router in main.py**

**Step 4: Commit**

```bash
git add packages/api/app/routers/batch.py packages/api/app/models/batch.py packages/api/app/main.py
git commit -m "feat: add batch API endpoints"
```

---

## Task 5: Backend - Multi-GPU Job Distribution

**Files:**
- Modify: `packages/api/app/services/job_manager.py`

**Step 1: Add GPU tracking**

Add to JobManager:
- `_gpu_ids: list[int]` - Available GPU IDs from settings
- `_gpu_busy_until: dict[int, datetime]` - Estimated completion time per GPU
- `_gpu_current_job: dict[int, str]` - Current job per GPU

**Step 2: Implement least-busy-first scheduling**

- `get_least_busy_gpu()` - Returns GPU that will be free soonest
- `estimate_job_duration(file_path)` - Rough estimate based on file size

**Step 3: Update job processing**

- Start one processor task per GPU
- Each processor pulls from shared queue
- Track GPU assignment in job status

**Step 4: Commit**

```bash
git add packages/api/app/services/job_manager.py
git commit -m "feat: add multi-GPU job distribution with least-busy-first scheduling"
```

---

## Task 6: Backend - S3 Upload Scaffold and Cleanup

**Files:**
- Create: `packages/api/app/services/s3_upload.py`
- Modify: `packages/api/app/services/pipeline_runner.py`

**Step 1: Create S3 scaffold**

Functions:
- `upload_results_to_s3()` - Scaffolded, logs files that would be uploaded
- `cleanup_job_files()` - Deletes input MP3 and output directory

**Step 2: Integrate into pipeline_runner.py**

After successful transcription:
1. Call S3 upload (scaffolded)
2. Call cleanup to delete local files

**Step 3: Commit**

```bash
git add packages/api/app/services/s3_upload.py packages/api/app/services/pipeline_runner.py
git commit -m "feat: add S3 upload scaffold and file cleanup"
```

---

## Task 7: Frontend - Multi-File Upload State Management

**Files:**
- Create: `packages/frontend/src/lib/upload.ts`

**Step 1: Create upload utilities**

Types:
- `UploadFile` - File with id, status, progress, jobId, error
- `UploadProgress` - Overall progress tracking

Functions:
- `uploadFileWithProgress()` - XHR upload with onprogress callback
- `uploadFilesWithProgress()` - Concurrent upload with configurable limit
- `extractFilesFromDataTransfer()` - Handle folder drops via webkitGetAsEntry
- `generateFileId()` - Unique ID for tracking

**Step 2: Commit**

```bash
git add packages/frontend/src/lib/upload.ts
git commit -m "feat: add multi-file upload utilities with progress tracking"
```

---

## Task 8: Frontend - Multi-File Upload UI Component

**Files:**
- Rewrite: `packages/frontend/src/components/JobSubmission.tsx`

**Step 1: Replace with multi-file version**

Features:
- Folder mode toggle (webkitdirectory)
- Multi-file drag-drop zone
- File queue with status icons (pending/uploading/complete/error)
- Overall progress bar with byte counts
- Expandable/collapsible file details
- Remove individual files before upload
- Clear all button
- Concurrent upload (3 files at a time)

**Step 2: Commit**

```bash
git add packages/frontend/src/components/JobSubmission.tsx
git commit -m "feat: rewrite JobSubmission with multi-file upload and progress UI"
```

---

## Task 9: Frontend - Add Progress Component (if missing)

**Files:**
- Check/Create: `packages/frontend/src/components/ui/progress.tsx`

**Step 1: Check if Progress component exists**

If not, create using @radix-ui/react-progress.

**Step 2: Install dependency if needed**

```bash
cd packages/frontend && npm install @radix-ui/react-progress
```

**Step 3: Commit**

```bash
git add packages/frontend/src/components/ui/progress.tsx packages/frontend/package.json packages/frontend/package-lock.json
git commit -m "feat: add Progress UI component"
```

---

## Task 10: Integration Testing

**Manual testing steps:**

1. **Start dev server:** `python scripts/dev.py`

2. **Test single file upload:**
   - Drag single .wav file
   - Verify progress bar
   - Verify job created

3. **Test multi-file upload:**
   - Select 3-5 files
   - Verify concurrent progress
   - Verify batch completes

4. **Test folder upload:**
   - Enable folder toggle
   - Drag folder with audio files
   - Verify extraction and upload

5. **Test FFmpeg conversion:**
   - Upload .wav or .mp4
   - Check server logs for conversion
   - Verify job succeeds

6. **Final commit:**

```bash
git add -A
git commit -m "feat: complete multi-file upload implementation"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | FFmpeg conversion utility | `services/ffmpeg.py` |
| 2 | Integrate FFmpeg in upload | `routers/files.py` |
| 3 | Batch model and schema | `schemas/batch.py`, `schemas/job.py` |
| 4 | Batch API endpoints | `routers/batch.py`, `models/batch.py` |
| 5 | Multi-GPU distribution | `services/job_manager.py` |
| 6 | S3 scaffold and cleanup | `services/s3_upload.py`, `services/pipeline_runner.py` |
| 7 | Upload utilities | `lib/upload.ts` |
| 8 | Multi-file UI component | `components/JobSubmission.tsx` |
| 9 | Progress component | `components/ui/progress.tsx` |
| 10 | Integration testing | Manual |
