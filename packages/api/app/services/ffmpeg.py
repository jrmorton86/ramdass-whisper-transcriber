"""
FFmpeg Conversion Utility - Converts audio/video files to MP3 format optimized for Whisper.

Uses subprocess.run in a thread for Windows compatibility.
"""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


async def convert_to_mp3(
    input_path: str | Path,
    output_path: str | Path,
) -> tuple[bool, Optional[str]]:
    """
    Convert an audio/video file to MP3 format optimized for Whisper transcription.

    Output is 16kHz mono MP3, which is the optimal format for Whisper.

    Args:
        input_path: Path to the input audio/video file
        output_path: Path where the output MP3 file will be written

    Returns:
        Tuple of (success, error_message). On success, error_message is None.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Validate input file exists
    if not input_path.exists():
        return False, f"Input file not found: {input_path}"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting {input_path} -> {output_path}")

    # Build FFmpeg command for Whisper-optimized output:
    # -i: input file
    # -vn: disable video
    # -ar 16000: 16kHz sample rate (Whisper native)
    # -ac 1: mono audio
    # -b:a 64k: 64kbps bitrate (sufficient for speech)
    # -y: overwrite output without asking
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "64k",
        "-y",
        str(output_path),
    ]

    logger.info(f"FFmpeg command: {' '.join(cmd)}")

    try:
        # Use subprocess.run in a thread for Windows compatibility
        def run_ffmpeg():
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

        result = await asyncio.to_thread(run_ffmpeg)

        if result.returncode != 0:
            error_message = result.stderr.strip() if result.stderr else ""
            # Extract the most relevant error line
            error_lines = [
                line for line in error_message.split("\n")
                if line.strip() and not line.startswith("ffmpeg version")
            ]
            short_error = error_lines[-1] if error_lines else "FFmpeg conversion failed"
            logger.error(f"FFmpeg conversion failed (code {result.returncode}): {short_error}")
            logger.error(f"Full stderr: {error_message}")
            return False, short_error

        # Verify output file was created
        if not output_path.exists():
            return False, "FFmpeg completed but output file was not created"

        logger.info(f"Successfully converted {input_path.name} to MP3")
        return True, None

    except FileNotFoundError:
        logger.error("FFmpeg not found in PATH")
        return False, "FFmpeg not found. Please ensure FFmpeg is installed and in PATH."
    except Exception as e:
        logger.exception(f"Unexpected error during FFmpeg conversion: {type(e).__name__}: {e}")
        error_msg = str(e) if str(e) else type(e).__name__
        return False, f"Conversion error: {error_msg}"


async def get_audio_duration(file_path: str | Path) -> Optional[float]:
    """
    Get the duration of an audio/video file in seconds.

    Uses FFprobe (part of FFmpeg) to extract duration metadata.
    Useful for job time estimation.

    Args:
        file_path: Path to the audio/video file

    Returns:
        Duration in seconds, or None if duration cannot be determined.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"File not found for duration check: {file_path}")
        return None

    # Use ffprobe to get duration
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.warning(f"FFprobe failed for {file_path.name}: {stderr.decode('utf-8', errors='replace').strip()}")
            return None

        duration_str = stdout.decode("utf-8", errors="replace").strip()

        if not duration_str:
            logger.warning(f"No duration found for {file_path.name}")
            return None

        duration = float(duration_str)
        logger.debug(f"Duration of {file_path.name}: {duration:.2f}s")
        return duration

    except FileNotFoundError:
        logger.warning("FFprobe not found. Please ensure FFmpeg is installed and in PATH.")
        return None
    except ValueError as e:
        logger.warning(f"Could not parse duration for {file_path.name}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error getting duration for {file_path.name}")
        return None
