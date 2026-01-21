"""
Logs API router - SSE streaming for real-time job logs and dashboard updates.
Supports both per-job log streaming and dashboard-wide event streaming.
"""

import json
import asyncio
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from ..services.job_manager import job_manager

router = APIRouter()


@router.get("/dashboard/stream")
async def stream_dashboard():
    """
    Stream dashboard updates via Server-Sent Events (SSE).

    Events:
    - 'job_created': New job was created
    - 'job_updated': Job status changed
    - 'job_progress': Job progress updated
    - 'ping': Keepalive (every 30s)
    """

    async def event_generator():
        # Subscribe to dashboard updates
        queue = job_manager.subscribe_dashboard()

        try:
            while True:
                try:
                    # Wait for event with timeout for keepalive
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)

                    if event is None:
                        break

                    # Determine event type
                    event_type = event.get("type", "update")

                    yield {
                        "event": event_type,
                        "data": json.dumps(event),
                    }

                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {
                        "event": "ping",
                        "data": "",
                    }

        except asyncio.CancelledError:
            # Client disconnected
            pass
        finally:
            # Unsubscribe when done
            job_manager.unsubscribe_dashboard(queue)

    return EventSourceResponse(event_generator())


@router.get("/jobs/{job_id}/logs")
async def stream_logs(job_id: str):
    """
    Stream job logs via Server-Sent Events (SSE).

    Events:
    - 'log': Log entry with timestamp, level, message, and ansi
    - 'progress': Progress update with stage and percentage
    - 'status': Job status change
    - 'ping': Keepalive (every 30s)
    - 'complete': Job finished (success or failure)
    """

    async def event_generator():
        # Get or create log queue for this job
        queue = job_manager.get_log_queue(job_id)

        if queue is None:
            # Job doesn't exist or already completed
            yield {
                "event": "error",
                "data": json.dumps({"message": "Job not found or already completed"}),
            }
            return

        try:
            while True:
                try:
                    # Wait for log entry with timeout for keepalive
                    log_entry = await asyncio.wait_for(queue.get(), timeout=30.0)

                    if log_entry is None:
                        # Job completed
                        yield {
                            "event": "complete",
                            "data": json.dumps({"jobId": job_id}),
                        }
                        break

                    # Determine event type
                    event_type = log_entry.get("type", "log")

                    yield {
                        "event": event_type,
                        "data": json.dumps(log_entry),
                    }

                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {
                        "event": "ping",
                        "data": "",
                    }

        except asyncio.CancelledError:
            # Client disconnected
            pass

    return EventSourceResponse(event_generator())


@router.get("/jobs/{job_id}/logs/history")
async def get_log_history(job_id: str, limit: int = 100):
    """
    Get historical logs for a job.

    Returns the most recent logs (up to limit).
    """
    logs = job_manager.get_log_history(job_id, limit=limit)

    if logs is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"logs": logs, "total": len(logs)}
