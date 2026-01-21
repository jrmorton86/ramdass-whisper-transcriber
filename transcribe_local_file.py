#!/usr/bin/env python3
"""
Local File Transcription Pipeline

Transcribes a local audio/video file and generates:
- Whisper transcript
- Refined Claude transcript
- Audio embeddings
- SRT file

All outputs are saved in a JSON file next to the original file.
No database or S3 interaction required.

Usage:
    python transcribe_local_file.py path/to/audio.mp3
    python transcribe_local_file.py path/to/video.mp4 --model large
"""

import sys
import argparse
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_transcription_pipeline(audio_path: Path, model: str = "medium", device: str | None = None) -> tuple:
    """
    Run the complete transcription pipeline:
    1. Whisper transcription (vocabulary-enhanced)
    2. Format with intelligent paragraphs + SRT
    3. Claude refinement with corrections JSON
    4. Apply corrections to SRT
    
    Returns tuple: (whisper_data, formatted_text, refined_text, segments, changes_json)
    """
    logger.info("🎤 Step 1/5: Running complete transcription pipeline...")
    
    venv_python = Path(__file__).parent / "venv" / "Scripts" / "python.exe"
    pipeline_script = Path(__file__).parent / "transcribe_pipeline" / "transcribe_pipeline.py"
    
    # Create temporary output directory
    tmp_dir = Path(tempfile.gettempdir()) / f"transcribe_{audio_path.stem}"
    tmp_dir.mkdir(exist_ok=True)
    
    try:
        cmd = [
            str(venv_python),
            str(pipeline_script),
            str(audio_path),
            "--model", model,
            "--output-dir", str(tmp_dir),
            "--silent"  # Silent mode for cleaner output
        ]
        
        if device:
            cmd.extend(["--device", device])
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if result.returncode != 0:
            logger.error(f"❌ Pipeline failed with exit code: {result.returncode}")
            if result.stdout:
                logger.error(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                logger.error(f"STDERR:\n{result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        
        # Read all output files
        base_name = audio_path.stem
        json_file = tmp_dir / f"{base_name}.json"
        formatted_file = tmp_dir / f"{base_name}_formatted.txt"
        refined_file = tmp_dir / f"{base_name}_formatted_refined.txt"
        changes_file = tmp_dir / f"{base_name}_formatted_refined_changes.json"
        
        if not json_file.exists():
            raise FileNotFoundError(f"Whisper JSON not found: {json_file}")
        
        # Read whisper data
        with open(json_file, 'r', encoding='utf-8') as f:
            whisper_data = json.load(f)
        
        # Read formatted text (if exists)
        formatted_text = None
        if formatted_file.exists():
            with open(formatted_file, 'r', encoding='utf-8') as f:
                formatted_text = f.read()
        
        # Read refined text (if exists)
        refined_text = None
        if refined_file.exists():
            with open(refined_file, 'r', encoding='utf-8') as f:
                refined_text = f.read()
        
        # Read changes JSON (if exists)
        changes_json = None
        if changes_file.exists():
            with open(changes_file, 'r', encoding='utf-8') as f:
                changes_json = json.load(f)
        
        logger.info("✅ Transcription pipeline complete")
        
        return whisper_data, formatted_text, refined_text, changes_json
        
    except subprocess.CalledProcessError:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise
    finally:
        # Cleanup temp directory
        if tmp_dir.exists():
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


def generate_embeddings(refined_text: str) -> list:
    """
    Generate embeddings for the refined transcript.
    Returns list of embedding vectors.
    """
    logger.info("🧮 Step 3/4: Generating embeddings...")
    
    venv_python = Path(__file__).parent / "venv" / "Scripts" / "python.exe"
    
    # Create temporary input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
        tmp.write(refined_text)
        tmp_input = Path(tmp.name)
    
    try:
        cmd = [
            str(venv_python),
            "-c",
            f"""
import sys
sys.path.insert(0, r'{Path(__file__).parent}')
from comprehend_utils.generate_audio_embeddings import generate_audio_embeddings

with open(r'{tmp_input}', 'r', encoding='utf-8') as f:
    text = f.read()

embeddings = generate_audio_embeddings(text)
import json
print(json.dumps(embeddings))
"""
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        embeddings = json.loads(result.stdout)
        
        logger.info(f"✅ Generated {len(embeddings)} embedding chunks")
        return embeddings
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse embeddings output: {e}")
        raise
    finally:
        if tmp_input.exists():
            tmp_input.unlink()


def generate_srt(segments: list, output_path: Path):
    """
    Generate SRT subtitle file from Whisper segments.
    """
    logger.info("📝 Step 4/4: Generating SRT file...")
    
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, segment in enumerate(segments, 1):
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            text = segment['text'].strip()
            
            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")
    
    logger.info(f"✅ SRT file created: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Transcribe a local audio/video file and save all outputs to JSON'
    )
    parser.add_argument('file', type=str,
                        help='Path to audio/video file to transcribe')
    parser.add_argument('--model', type=str, default='medium',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size (default: medium)')
    parser.add_argument('-d', '--device', type=str,
                        help='CUDA device to use (e.g., cuda:0, cuda:1)')
    parser.add_argument('--skip-embeddings', action='store_true',
                        help='Skip embedding generation step')
    args = parser.parse_args()
    
    # Resolve file path
    file_path = Path(args.file).resolve()
    
    if not file_path.exists():
        logger.error(f"❌ File not found: {file_path}")
        return 1
    
    logger.info("=" * 80)
    logger.info("LOCAL FILE TRANSCRIPTION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"File: {file_path.name}")
    logger.info(f"Model: {args.model}")
    if args.device:
        logger.info(f"Device: {args.device}")
    logger.info("=" * 80)
    logger.info("")
    
    # Prepare output paths
    output_json = file_path.with_suffix('.json')
    output_srt = file_path.with_suffix('.srt')
    
    try:
        # Step 1-4: Complete transcription pipeline
        # This runs: Whisper -> Format with Paragraphs -> Claude Refinement -> Apply Corrections to SRT
        whisper_data, formatted_text, refined_text, changes_json = run_transcription_pipeline(file_path, args.model, args.device)
        transcript_text = whisper_data.get('text', '')
        segments = whisper_data.get('segments', [])
        
        # Step 5: Generate embeddings (optional)
        embeddings = None
        if not args.skip_embeddings:
            try:
                # Use refined text if available, otherwise formatted text, otherwise raw transcript
                text_for_embeddings = refined_text or formatted_text or transcript_text
                embeddings = generate_embeddings(text_for_embeddings)
            except Exception as e:
                logger.warning(f"⚠️  Embedding generation failed, continuing without it: {e}")
        
        # Save formatted and refined text files alongside the media file
        if formatted_text:
            formatted_output = Path(str(file_path.with_suffix('')) + '_formatted.txt')
            with open(formatted_output, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            logger.info(f"📄 Saved formatted text: {formatted_output.name}")
        
        if refined_text:
            refined_output = Path(str(file_path.with_suffix('')) + '_refined.txt')
            with open(refined_output, 'w', encoding='utf-8') as f:
                f.write(refined_text)
            logger.info(f"📄 Saved refined text: {refined_output.name}")
        
        if changes_json:
            changes_output = Path(str(file_path.with_suffix('')) + '_changes.json')
            with open(changes_output, 'w', encoding='utf-8') as f:
                json.dump(changes_json, f, indent=2, ensure_ascii=False)
            logger.info(f"📄 Saved changes JSON: {changes_output.name}")
        
        # Create output JSON
        output_data = {
            "metadata": {
                "filename": file_path.name,
                "file_path": str(file_path),
                "processed_at": datetime.utcnow().isoformat(),
                "whisper_model": args.model,
                "device": args.device
            },
            "whisper_transcript": {
                "text": transcript_text,
                "segments": segments,
                "language": whisper_data.get('language'),
            },
            "formatted_text": formatted_text,
            "refined_transcript": refined_text,
            "claude_changes": changes_json,
            "srt_file": str(output_srt)
        }
        
        # Save JSON
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ TRANSCRIPTION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"JSON output: {output_json.name}")
        logger.info(f"SRT output: {output_srt.name}")
        logger.info(f"Transcript length: {len(transcript_text):,} characters")
        if refined_text:
            logger.info(f"Refined length: {len(refined_text):,} characters")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
