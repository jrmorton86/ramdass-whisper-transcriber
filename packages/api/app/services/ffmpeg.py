"""
FFmpeg Conversion Utility - Converts audio/video files to MP3 format optimized for Whisper.

Uses asyncio.create_subprocess_exec which is safe from shell injection
as it does not use a shell and passes arguments directly.
"""

import asyncio
import logging
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

    try:
        # Use asyncio.create_subprocess_exec for safe subprocess execution
        # This avoids shell injection as arguments are passed directly
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        _, stderr = await process.communicate()

        if process.returncode != 0:
            error_message = stderr.decode("utf-8", errors="replace").strip()
            # Extract the most relevant error line
            error_lines = [
                line for line in error_message.split("\n")
                if line.strip() and not line.startswith("ffmpeg version")
            ]
            short_error = error_lines[-1] if error_lines else "FFmpeg conversion failed"
            logger.error(f"FFmpeg conversion failed: {short_error}")
            return False, short_error

        # Verify output file was created
        if not output_path.exists():
            return False, "FFmpeg completed but output file was not created"

        logger.info(f"Successfully converted {input_path.name} to MP3")
        return True, None

    except FileNotFoundError:
        return False, "FFmpeg not found. Please ensure FFmpeg is installed and in PATH."
    except Exception as e:
        logger.exception("Unexpected error during FFmpeg conversion")
        return False, f"Conversion error: {str(e)}"


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
