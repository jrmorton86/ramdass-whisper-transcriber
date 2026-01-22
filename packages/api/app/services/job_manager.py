"""
Job Manager - Handles background job processing and log streaming.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional
from collections import defaultdict

from .pipeline_runner import PipelineRunner

logger = logging.getLogger(__name__)


class JobManager:
    """
    Manages job queue, background processing, and log streaming.
    """

    # Define pipeline steps in order
    # For local file processing: Transcribing -> Formatting -> Refining -> Post-processing -> Complete
    # For UUID processing: adds Downloading (step 1) and Uploading (step 6)
    PIPELINE_STEPS = [
        "Transcribing",     # 1 - Whisper transcription
        "Formatting",       # 2 - Convert to SRT + formatted text
        "Refining",         # 3 - Claude refinement
        "Post-processing",  # 4 - Apply corrections to SRT
        "Complete",         # 5 - Pipeline done
    ]
    TOTAL_STEPS = len(PIPELINE_STEPS)

    def __init__(self):
        self.pending_jobs: asyncio.Queue = asyncio.Queue()
        self.active_jobs: dict[str, dict] = {}
        self.cancelled_jobs: set[str] = set()

        # Log queues for SSE streaming (job_id -> queue)
        self.log_queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)

        # Log history for completed/historical access
        self.log_history: dict[str, list] = defaultdict(list)

        # Dashboard subscribers for real-time updates (set of queues)
        self._dashboard_subscribers: set[asyncio.Queue] = set()

        # Background task reference
        self._processor_task: Optional[asyncio.Task] = None
        self._running = False

    def subscribe_dashboard(self) -> asyncio.Queue:
        """Subscribe to dashboard updates. Returns a queue to listen on."""
        queue = asyncio.Queue()
        self._dashboard_subscribers.add(queue)
        logger.info(f"Dashboard subscriber added. Total: {len(self._dashboard_subscribers)}")
        return queue

    def unsubscribe_dashboard(self, queue: asyncio.Queue):
        """Unsubscribe from dashboard updates."""
        self._dashboard_subscribers.discard(queue)
        logger.info(f"Dashboard subscriber removed. Total: {len(self._dashboard_subscribers)}")

    async def _broadcast_dashboard(self, event_type: str, data: dict):
        """Broadcast an event to all dashboard subscribers."""
        if not self._dashboard_subscribers:
            return

        event = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            **data,
        }

        # Send to all subscribers (remove any that fail)
        failed = []
        for queue in self._dashboard_subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                failed.append(queue)

        for queue in failed:
            self._dashboard_subscribers.discard(queue)

    async def start(self):
        """Start the background job processor."""
        self._running = True
        self._processor_task = asyncio.create_task(self._process_jobs())
        logger.info("Job manager started")

    async def stop(self):
        """Stop the background job processor."""
        self._running = False

        # Cancel all pending jobs
        for job_id in list(self.active_jobs.keys()):
            await self.cancel_job(job_id)

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        logger.info("Job manager stopped")

    async def submit_job(self, job_id: str, job_type: str, input_data: str):
        """Submit a job for processing."""
        self.active_jobs[job_id] = {
            "type": job_type,
            "input": input_data,
            "status": "pending",
            "progress": 0,
        }

        # Create log queue
        self.log_queues[job_id] = asyncio.Queue()
        self.log_history[job_id] = []

        await self.pending_jobs.put({
            "id": job_id,
            "type": job_type,
            "input": input_data,
        })

        # Broadcast to dashboard subscribers
        await self._broadcast_dashboard("job_created", {"jobId": job_id})

        logger.info(f"Job {job_id} submitted for processing")

    async def cancel_job(self, job_id: str):
        """Cancel a job."""
        self.cancelled_jobs.add(job_id)

        # Signal completion to log listeners
        if job_id in self.log_queues:
            await self.log_queues[job_id].put(None)

        logger.info(f"Job {job_id} cancelled")

    def get_log_queue(self, job_id: str) -> Optional[asyncio.Queue]:
        """Get the log queue for a job."""
        return self.log_queues.get(job_id)

    def get_log_history(self, job_id: str, limit: int = 100) -> Optional[list]:
        """Get historical logs for a job."""
        if job_id not in self.log_history:
            return None
        return self.log_history[job_id][-limit:]

    async def _emit_log(self, job_id: str, level: str, message: str, **extra):
        """Emit a log entry to all listeners."""
        log_entry = {
            "type": "log",
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            "ansi": message,  # Preserve ANSI codes
            **extra,
        }

        # Add to history
        self.log_history[job_id].append(log_entry)

        # Send to queue
        if job_id in self.log_queues:
            await self.log_queues[job_id].put(log_entry)

    async def _emit_progress(self, job_id: str, progress: int, stage: str = None, step: int = None):
        """Emit a progress update."""
        self.active_jobs[job_id]["progress"] = progress
        if stage:
            self.active_jobs[job_id]["stage"] = stage
        if step is not None:
            self.active_jobs[job_id]["step"] = step

        progress_entry = {
            "type": "progress",
            "timestamp": datetime.utcnow().isoformat(),
            "progress": progress,
            "stage": stage,
            "step": step,
            "totalSteps": self.TOTAL_STEPS,
        }

        # Add to history
        self.log_history[job_id].append(progress_entry)

        # Send to queue
        if job_id in self.log_queues:
            await self.log_queues[job_id].put(progress_entry)

        # Broadcast to dashboard subscribers
        await self._broadcast_dashboard("job_progress", {
            "jobId": job_id,
            "progress": progress,
            "stage": stage,
            "step": step,
            "totalSteps": self.TOTAL_STEPS,
        })

    async def _emit_status(self, job_id: str, status: str, error: str = None):
        """Emit a status change."""
        status_entry = {
            "type": "status",
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
            "error": error,
        }

        # Add to history
        self.log_history[job_id].append(status_entry)

        # Send to queue
        if job_id in self.log_queues:
            await self.log_queues[job_id].put(status_entry)

        # Broadcast to dashboard subscribers
        await self._broadcast_dashboard("job_updated", {
            "jobId": job_id,
            "status": status,
            "error": error,
        })

    async def _process_jobs(self):
        """Background task to process job queue."""
        while self._running:
            try:
                # Wait for a job
                job = await asyncio.wait_for(
                    self.pending_jobs.get(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            job_id = job["id"]

            # Check if cancelled
            if job_id in self.cancelled_jobs:
                self.cancelled_jobs.discard(job_id)
                continue

            try:
                # Update status
                self.active_jobs[job_id]["status"] = "processing"
                await self._emit_status(job_id, "processing")

                # Process the job
                await self._run_job(job)

            except Exception as e:
                logger.exception(f"Error processing job {job_id}")
                await self._emit_log(job_id, "error", f"Job failed: {str(e)}")
                await self._emit_status(job_id, "failed", str(e))
                await self._update_job_in_db(job_id, "failed", error=str(e))

            finally:
                # Signal completion to log listeners
                if job_id in self.log_queues:
                    await self.log_queues[job_id].put(None)

                # Clean up
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]

    async def _run_job(self, job: dict):
        """Run a single job through the pipeline."""
        job_id = job["id"]
        job_type = job["type"]
        input_data = job["input"]

        # Create pipeline runner with log callback
        async def log_callback(entry: dict):
            await self._emit_log(
                job_id,
                entry.get("level", "info"),
                entry.get("message", ""),
            )

            # Parse progress from log if available
            progress = self._parse_progress(entry.get("message", ""))
            if progress:
                stage, percentage, step = progress
                await self._emit_progress(job_id, percentage, stage, step)

        runner = PipelineRunner(log_callback)

        # Mark started
        await self._update_job_in_db(job_id, "processing", started=True)

        # Check cancellation periodically
        def is_cancelled():
            return job_id in self.cancelled_jobs

        # Run the appropriate pipeline
        if job_type == "uuid":
            result = await runner.process_uuid(job_id, input_data, is_cancelled)
        else:
            result = await runner.process_file(job_id, input_data, is_cancelled)

        # Check if cancelled during processing
        if job_id in self.cancelled_jobs:
            self.cancelled_jobs.discard(job_id)
            return

        # Get logs for this job to save to database
        job_logs = self.log_history.get(job_id, [])

        # Update final status
        if result["success"]:
            await self._emit_status(job_id, "completed")
            await self._update_job_in_db(
                job_id,
                "completed",
                result_text=result.get("text"),
                result_segments=result.get("segments"),
                result_metadata=result.get("metadata"),
                logs=job_logs,
            )
        else:
            await self._emit_status(job_id, "failed", result.get("error"))
            await self._update_job_in_db(job_id, "failed", error=result.get("error"), logs=job_logs)

    def _parse_progress(self, message: str) -> Optional[tuple[str, int, int]]:
        """Parse progress information from log message.

        Returns: (stage_name, progress_percentage, step_number) or None

        Steps (5 total for local files):
        1. Transcribing - Whisper transcription
        2. Formatting - Convert to SRT + formatted text
        3. Refining - Claude refinement
        4. Post-processing - Apply corrections to SRT
        5. Complete - Pipeline done

        Note: Order matters! More specific matches should come first.
        Avoid matching "completed successfully" as pipeline complete.
        """
        lower = message.lower()

        # Skip generic completion messages (e.g., "completed successfully")
        # These are stage completions, not pipeline completion
        if "completed successfully" in lower:
            return None

        # Match specific [STEP] markers from transcribe_pipeline.py
        if "[step]" in lower:
            if "transcribing" in lower:
                return ("Transcribing", 10, 1)
            elif "formatting" in lower:
                return ("Formatting", 40, 2)
            elif "refining" in lower:
                return ("Refining", 55, 3)
            elif "post-processing" in lower:
                return ("Post-processing", 80, 4)

        # Match stage headers (STAGE: ...)
        if "stage:" in lower:
            if "whisper" in lower:
                return ("Transcribing", 15, 1)
            elif "format" in lower:
                return ("Formatting", 45, 2)
            elif "claude" in lower:
                return ("Refining", 60, 3)
            elif "srt" in lower or "corrections" in lower:
                return ("Post-processing", 85, 4)

        # Final pipeline completion - only match the specific message
        if "pipeline complete" in lower or "[ok] pipeline complete" in lower:
            return ("Complete", 100, 5)

        return None

    async def _update_job_in_db(
        self,
        job_id: str,
        status: str,
        started: bool = False,
        error: str = None,
        result_text: str = None,
        result_segments: list = None,
        result_metadata: dict = None,
        logs: list = None,
    ):
        """Update job in database."""
        from ..database import async_session_maker
        from ..schemas.job import Job

        async with async_session_maker() as session:
            from sqlalchemy import select

            result = await session.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()

            if job:
                job.status = status

                if started:
                    job.started_at = datetime.utcnow()

                if status in ["completed", "failed", "cancelled"]:
                    job.completed_at = datetime.utcnow()
                    if job.started_at:
                        job.duration = (job.completed_at - job.started_at).total_seconds()

                if error:
                    job.error = error

                if result_text:
                    job.result_text = result_text

                if result_segments:
                    job.result_segments = json.dumps(result_segments)

                if result_metadata:
                    job.result_metadata = json.dumps(result_metadata)

                if logs is not None:
                    job.logs = json.dumps(logs)

                await session.commit()


# Global job manager instance
job_manager = JobManager()
