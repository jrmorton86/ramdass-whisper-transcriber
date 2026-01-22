"""
Pipeline Runner - Runs transcription pipeline with real-time log capture.

Uses subprocess.Popen for cross-platform compatibility (Windows/Linux).
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from ..config import settings
from .s3_upload import upload_results_to_s3, cleanup_job_files


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
        gpu_id: Optional[int] = None,
    ) -> dict:
        """
        Process an Intelligence Bank UUID through the pipeline.

        Args:
            job_id: Job ID for tracking
            asset_uuid: The asset UUID to process
            is_cancelled: Function to check if job is cancelled
            gpu_id: Optional GPU ID to use for processing

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

            # Now process the file with the same GPU ID
            return await self.process_file(job_id, str(audio_path), is_cancelled, gpu_id=gpu_id)

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
        gpu_id: Optional[int] = None,
    ) -> dict:
        """
        Process an audio file through the transcription pipeline.

        Args:
            job_id: Job ID for tracking
            file_path: Path to the audio file
            is_cancelled: Function to check if job is cancelled
            gpu_id: Optional GPU ID to use for processing

        Returns:
            dict with success status and result/error
        """
        # Convert to absolute path (file_path may be relative to API working dir)
        abs_file_path = Path(file_path).resolve()
        await self._log("info", f"Starting pipeline for file: {abs_file_path}")

        # Verify file exists
        if not abs_file_path.exists():
            return {"success": False, "error": f"Audio file not found: {abs_file_path}"}

        # Get the transcribe_pipeline.py script
        pipeline_script = self.pipeline_dir / "transcribe_pipeline" / "transcribe_pipeline.py"

        if not pipeline_script.exists():
            return {"success": False, "error": f"Pipeline script not found: {pipeline_script}"}

        # Create output directory (absolute path)
        output_dir = (Path.cwd() / "tmp" / job_id).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine GPU to use (explicit gpu_id or first from settings)
        effective_gpu_id = gpu_id if gpu_id is not None else int(settings.gpu_ids.split(",")[0])
        await self._log("info", f"Using GPU {effective_gpu_id} for transcription")

        # Build command - use pipeline Python (has whisper installed)
        # Note: Always use cuda:0 since CUDA_VISIBLE_DEVICES handles actual GPU selection
        cmd = [
            str(settings.pipeline_python),
            str(pipeline_script),
            str(abs_file_path),
            "--model", settings.whisper_model,
            "--output-dir", str(output_dir),
            "--device", "cuda:0",
        ]

        # Run the pipeline using safe subprocess execution with GPU environment
        # CUDA_VISIBLE_DEVICES will be set to effective_gpu_id, making physical GPU N appear as cuda:0
        returncode = await self._run_subprocess(cmd, str(self.pipeline_dir), is_cancelled, gpu_id=effective_gpu_id)

        if is_cancelled():
            return {"success": False, "error": "Job cancelled"}

        if returncode != 0:
            return {"success": False, "error": f"Pipeline failed with code {returncode}"}

        # Read results - pipeline outputs simplified to 3 files:
        # {base}.txt (refined transcript), {base}.srt (refined subtitles), {base}.json (whisper output)
        base_name = Path(file_path).stem
        transcript_txt = output_dir / f"{base_name}.txt"
        srt_file = output_dir / f"{base_name}.srt"
        json_file = output_dir / f"{base_name}.json"

        await self._log("info", f"Looking for transcript files with base: {base_name}")
        await self._log("info", f"  TXT: {transcript_txt.exists()}, SRT: {srt_file.exists()}, JSON: {json_file.exists()}")

        # Check for transcript file
        if not transcript_txt.exists():
            return {"success": False, "error": f"Transcript file not found: {transcript_txt}"}

        await self._log("info", "Found transcript file")

        # Read transcript
        with open(transcript_txt, "r", encoding="utf-8") as f:
            text = f.read()

        # Read segments from SRT if available, otherwise from JSON
        segments = []
        if srt_file.exists():
            await self._log("info", "Loading segments from SRT file")
            segments = self._parse_srt(srt_file)
        elif json_file.exists():
            await self._log("info", "Loading segments from Whisper JSON")
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

        # Upload results to S3 (scaffolded) and cleanup local files
        s3_result = await upload_results_to_s3(job_id, output_dir)
        if s3_result["success"]:
            await self._log("info", f"S3 upload scaffold completed for {len(s3_result['files'])} files")
            # Only cleanup if S3 "upload" succeeded
            cleanup_success = await cleanup_job_files(job_id, abs_file_path, output_dir)
            if cleanup_success:
                await self._log("info", "Local file cleanup completed")
            else:
                await self._log("warning", "Local file cleanup had some failures")
        else:
            await self._log("warning", "S3 upload scaffold failed, skipping cleanup")

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
        gpu_id: Optional[int] = None,
    ) -> int:
        """
        Run a subprocess with real-time log capture.

        Uses subprocess.Popen for Windows compatibility, with async log reading.
        Handles both newline-terminated output and carriage-return progress updates
        (like tqdm progress bars used by Whisper).

        Args:
            cmd: Command and arguments as a list
            cwd: Working directory
            is_cancelled: Function to check cancellation
            gpu_id: Optional GPU ID to use (sets CUDA_VISIBLE_DEVICES)

        Returns:
            Process return code
        """
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # Ensure unbuffered output

        # Set GPU visibility if specified
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Use Popen for Windows compatibility
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env,
        )

        # Track last progress percentage to avoid spamming logs
        last_progress_pct = -1

        def parse_progress(text: str) -> Optional[int]:
            """Extract progress percentage from PROGRESS: X% output."""
            # Match our custom "PROGRESS: 50%" format
            match = re.search(r'PROGRESS:\s*(\d+)%', text)
            if match:
                return int(match.group(1))
            return None

        # Read output handling both \n and \r terminated lines
        async def read_output():
            nonlocal last_progress_pct
            loop = asyncio.get_event_loop()
            buffer = ""

            while True:
                if is_cancelled():
                    process.terminate()
                    return

                # Read small chunks for responsive progress updates
                # 256 bytes is small enough for real-time feedback but not too slow
                chunk = await loop.run_in_executor(None, lambda: process.stdout.read(256))

                if not chunk:
                    # Process any remaining buffer
                    if buffer.strip():
                        level = self._detect_level(buffer)
                        await self._log(level, buffer.strip())
                    break

                # Decode and add to buffer
                buffer += chunk.decode("utf-8", errors="replace")

                # Process complete lines (split on \n or \r)
                while '\n' in buffer or '\r' in buffer:
                    # Find the first line terminator
                    newline_pos = buffer.find('\n')
                    cr_pos = buffer.find('\r')

                    if newline_pos == -1:
                        split_pos = cr_pos
                    elif cr_pos == -1:
                        split_pos = newline_pos
                    else:
                        split_pos = min(newline_pos, cr_pos)

                    line = buffer[:split_pos].strip()
                    buffer = buffer[split_pos + 1:]

                    if not line:
                        continue

                    # Check if this is a progress update
                    progress = parse_progress(line)
                    if progress is not None:
                        # Only log progress at certain intervals to avoid spam
                        if progress > last_progress_pct and (progress % 5 == 0 or progress == 100):
                            last_progress_pct = progress
                            await self._log("info", f"[PROGRESS] Whisper transcription: {progress}%")
                    else:
                        # Regular log line
                        level = self._detect_level(line)
                        await self._log(level, line)

        await read_output()
        process.wait()
        return process.returncode

    async def _run_subprocess_capture_output(
        self,
        cmd: list[str],
        is_cancelled: Callable[[], bool],
    ) -> tuple[int, str]:
        """
        Run a subprocess and capture its output.

        Uses subprocess.Popen for Windows compatibility.

        Returns:
            Tuple of (return_code, output)
        """
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        loop = asyncio.get_event_loop()

        # Run in executor to avoid blocking
        def run_process():
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            return process.communicate(), process.returncode

        (stdout, stderr), returncode = await loop.run_in_executor(None, run_process)

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

        return returncode, output

    def _parse_srt(self, srt_path: Path) -> list[dict]:
        """Parse SRT file into segments list.

        Args:
            srt_path: Path to the SRT file

        Returns:
            List of segment dicts with start, end, text keys
        """
        segments = []

        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by double newlines to get individual subtitle blocks
        blocks = content.strip().split("\n\n")

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                # Line 0: subtitle number (skip)
                # Line 1: timestamp (00:00:00,000 --> 00:00:05,000)
                # Line 2+: text
                timestamp_line = lines[1]
                text_lines = lines[2:]

                # Parse timestamp
                if " --> " in timestamp_line:
                    start_str, end_str = timestamp_line.split(" --> ")
                    start = self._srt_time_to_seconds(start_str.strip())
                    end = self._srt_time_to_seconds(end_str.strip())
                    text = " ".join(text_lines).strip()

                    segments.append({
                        "start": start,
                        "end": end,
                        "text": text,
                    })

        return segments

    def _srt_time_to_seconds(self, time_str: str) -> float:
        """Convert SRT timestamp to seconds.

        Args:
            time_str: Time in format HH:MM:SS,mmm

        Returns:
            Time in seconds as float
        """
        # Handle both comma and period as decimal separator
        time_str = time_str.replace(",", ".")

        parts = time_str.split(":")
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds

        return 0.0

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
