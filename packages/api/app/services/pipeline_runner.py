"""
Pipeline Runner - Runs transcription pipeline with real-time log capture.

Uses asyncio.create_subprocess_exec for safe subprocess execution (no shell injection).
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from ..config import settings


class PipelineRunner:
    """
    Runs the transcription pipeline as a subprocess with real-time log capture.

    Uses asyncio.create_subprocess_exec which is safe from shell injection
    as it does not use a shell and passes arguments directly.
    """

    def __init__(self, log_callback: Callable):
        """
        Initialize pipeline runner.

        Args:
            log_callback: Async function to call with log entries
        """
        self.log_callback = log_callback
        self.pipeline_dir = settings.pipeline_dir

    async def process_uuid(
        self,
        job_id: str,
        asset_uuid: str,
        is_cancelled: Callable[[], bool],
    ) -> dict:
        """
        Process an Intelligence Bank UUID through the pipeline.

        Args:
            job_id: Job ID for tracking
            asset_uuid: The asset UUID to process
            is_cancelled: Function to check if job is cancelled

        Returns:
            dict with success status and result/error
        """
        await self._log("info", f"Starting pipeline for UUID: {asset_uuid}")

        # Create temp directory for this job
        temp_dir = Path("tmp") / job_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Use the batch_process_audio.py script which handles UUID processing
            # Or we can use transcribe_local_file.py after downloading
            # For now, let's use a simpler approach with the main pipeline

            # First, download the asset using intelligencebank_utils
            await self._log("info", "Downloading audio from Intelligence Bank...")

            download_script = self.pipeline_dir / "intelligencebank_utils" / "download_asset.py"

            if not download_script.exists():
                return {"success": False, "error": "Download script not found"}

            # Download the asset - using safe subprocess execution
            download_cmd = [
                sys.executable,
                str(download_script),
                asset_uuid,
                "--output-dir", str(temp_dir),
            ]

            returncode, audio_path = await self._run_subprocess_capture_output(
                download_cmd,
                is_cancelled,
            )

            if returncode != 0:
                return {"success": False, "error": "Failed to download asset"}

            # Find the downloaded file
            audio_files = list(temp_dir.glob("*.mp3")) + list(temp_dir.glob("*.wav")) + list(temp_dir.glob("*.m4a"))
            if not audio_files:
                return {"success": False, "error": "No audio file found after download"}

            audio_path = audio_files[0]

            # Now process the file
            return await self.process_file(job_id, str(audio_path), is_cancelled)

        finally:
            # Cleanup temp directory
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    async def process_file(
        self,
        job_id: str,
        file_path: str,
        is_cancelled: Callable[[], bool],
    ) -> dict:
        """
        Process an audio file through the transcription pipeline.

        Args:
            job_id: Job ID for tracking
            file_path: Path to the audio file
            is_cancelled: Function to check if job is cancelled

        Returns:
            dict with success status and result/error
        """
        await self._log("info", f"Starting pipeline for file: {file_path}")

        # Get the transcribe_pipeline.py script
        pipeline_script = self.pipeline_dir / "transcribe_pipeline" / "transcribe_pipeline.py"

        if not pipeline_script.exists():
            return {"success": False, "error": f"Pipeline script not found: {pipeline_script}"}

        # Create output directory
        output_dir = Path("tmp") / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command - all arguments are passed as list items (safe)
        cmd = [
            sys.executable,
            str(pipeline_script),
            file_path,
            "--model", settings.whisper_model,
            "--output-dir", str(output_dir),
        ]

        # Add GPU device if specified
        gpu_ids = settings.gpu_ids.split(",")
        if gpu_ids:
            cmd.extend(["--device", f"cuda:{gpu_ids[0]}"])

        # Run the pipeline using safe subprocess execution
        returncode = await self._run_subprocess(cmd, str(self.pipeline_dir), is_cancelled)

        if is_cancelled():
            return {"success": False, "error": "Job cancelled"}

        if returncode != 0:
            return {"success": False, "error": f"Pipeline failed with code {returncode}"}

        # Read results
        base_name = Path(file_path).stem
        refined_txt = output_dir / f"{base_name}_formatted_refined.txt"
        json_file = output_dir / f"{base_name}.json"

        if not refined_txt.exists():
            return {"success": False, "error": "Refined transcript not found"}

        # Read transcript
        with open(refined_txt, "r", encoding="utf-8") as f:
            text = f.read()

        # Read segments from JSON if available
        segments = []
        if json_file.exists():
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                segments = [
                    {
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "text": seg.get("text", ""),
                    }
                    for seg in data.get("segments", [])
                ]

        await self._log("info", "Pipeline completed successfully!")

        return {
            "success": True,
            "text": text,
            "segments": segments,
            "metadata": {
                "duration": len(segments) and segments[-1].get("end", 0) or 0,
                "language": "en",
                "model": f"whisper-{settings.whisper_model}",
            },
        }

    async def _run_subprocess(
        self,
        cmd: list[str],
        cwd: str,
        is_cancelled: Callable[[], bool],
    ) -> int:
        """
        Run a subprocess with real-time log capture.

        Uses asyncio.create_subprocess_exec which is safe from shell injection
        as it executes the command directly without shell interpretation.

        Args:
            cmd: Command and arguments as a list
            cwd: Working directory
            is_cancelled: Function to check cancellation

        Returns:
            Process return code
        """
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # Ensure unbuffered output

        # create_subprocess_exec is safe - no shell, arguments passed directly
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=env,
        )

        # Read output line by line
        while True:
            if is_cancelled():
                process.terminate()
                await process.wait()
                return -1

            try:
                line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=0.1,
                )
            except asyncio.TimeoutError:
                # Check if process is still running
                if process.returncode is not None:
                    break
                continue

            if not line:
                break

            decoded = line.decode("utf-8", errors="replace").rstrip()
            if decoded:
                level = self._detect_level(decoded)
                await self._log(level, decoded)

        await process.wait()
        return process.returncode

    async def _run_subprocess_capture_output(
        self,
        cmd: list[str],
        is_cancelled: Callable[[], bool],
    ) -> tuple[int, str]:
        """
        Run a subprocess and capture its output.

        Uses asyncio.create_subprocess_exec for safe execution.

        Returns:
            Tuple of (return_code, output)
        """
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        stdout, stderr = await process.communicate()

        # Log any errors
        if stderr:
            for line in stderr.decode("utf-8", errors="replace").strip().split("\n"):
                if line:
                    await self._log("error", line)

        # Log stdout
        output = stdout.decode("utf-8", errors="replace").strip()
        for line in output.split("\n"):
            if line:
                await self._log("info", line)

        return process.returncode, output

    def _detect_level(self, message: str) -> str:
        """Detect log level from message content."""
        lower = message.lower()

        if "error" in lower or "fail" in lower or "exception" in lower:
            return "error"
        elif "warn" in lower:
            return "warning"
        elif "debug" in lower:
            return "debug"
        elif "[ok]" in lower or "success" in lower or "complete" in lower:
            return "info"

        return "info"

    async def _log(self, level: str, message: str):
        """Emit a log entry."""
        await self.log_callback({
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
        })
