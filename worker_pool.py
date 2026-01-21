#!/usr/bin/env python3
"""
Worker Pool for Batch Transcription

Persistent GPU workers that load Whisper model once and process multiple assets.
Eliminates model loading overhead from subprocess-per-asset architecture.
"""

import multiprocessing as mp
from multiprocessing import Process, Queue
import logging
import os
import sys
import json
import shutil
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(processName)-12s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TranscriptionWorker:
    """
    Persistent worker that holds Whisper model in GPU memory.

    Designed to process multiple assets without reloading the model.
    """

    def __init__(self, gpu_id: int, model_name: str = "medium"):
        """
        Initialize worker and load Whisper model.

        Args:
            gpu_id: GPU device ID (0, 1, etc.)
            model_name: Whisper model size
        """
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.model_name = model_name
        self.model = None
        self.vocab_data = None
        self.replacement_map = None

    def initialize(self):
        """Load model and vocabulary. Called once at worker startup."""
        import torch
        import whisper
        from transcribe_pipeline.whisper_with_vocab import load_vocabulary_data

        logger.info(f"Initializing worker on {self.device}")
        logger.info(f"Loading Whisper model: {self.model_name}")

        # Load model directly to GPU
        self.model = whisper.load_model(self.model_name, device=self.device)

        # Log memory usage
        vram_allocated = torch.cuda.memory_allocated(self.gpu_id) / 1024**3
        logger.info(f"[OK] Model loaded on {self.device} - VRAM: {vram_allocated:.2f} GB")

        # Load vocabulary
        self.vocab_data, self.replacement_map = load_vocabulary_data()

        logger.info(f"[OK] Worker initialized on {self.device}")

    def process_asset(self, asset_uuid: str, asset_name: str, temp_dir: str) -> Dict[str, Any]:
        """
        Process a single asset through the full pipeline.

        Args:
            asset_uuid: Asset UUID
            asset_name: Asset display name
            temp_dir: Temporary directory for this asset

        Returns:
            dict with status, paths, and any errors
        """
        import torch
        from transcribe_pipeline.whisper_with_vocab import (
            transcribe_audio, save_transcription_result
        )
        from transcribe_pipeline.convert_aws_transcribe import format_whisper_transcript
        from transcribe_pipeline.claude_refine_transcript import refine_transcript_text
        from transcribe_pipeline.apply_txt_corrections_to_srt import apply_corrections_to_srt_file
        from post_process_transcript import post_process_transcript
        from intelligencebank_utils.download_asset import download_asset

        result = {
            'uuid': asset_uuid,
            'name': asset_name,
            'status': 'failed',
            'error': None
        }

        temp_path = Path(temp_dir)

        try:
            temp_path.mkdir(parents=True, exist_ok=True)

            # Step 1: Download audio
            logger.info(f"[{asset_uuid}] Downloading audio...")
            audio_path = asyncio.run(download_asset(asset_uuid))

            # Move to temp dir
            audio_in_temp = temp_path / Path(audio_path).name
            shutil.move(str(audio_path), str(audio_in_temp))
            audio_path = audio_in_temp
            logger.info(f"[{asset_uuid}] Downloaded: {audio_path}")

            base_name = audio_path.stem
            output_base = temp_path / base_name

            # Step 2: Transcribe with pre-loaded model
            logger.info(f"[{asset_uuid}] Transcribing...")
            transcription = transcribe_audio(
                audio_path=str(audio_path),
                model=self.model,
                vocab_data=self.vocab_data,
                replacement_map=self.replacement_map,
                device=self.device
            )
            save_transcription_result(transcription, str(output_base))

            # Step 3: Format to SRT + TXT
            logger.info(f"[{asset_uuid}] Formatting...")
            format_result = format_whisper_transcript(
                transcript_json=transcription,
                output_dir=str(temp_path),
                base_filename=base_name
            )

            # Step 4: Claude refinement
            logger.info(f"[{asset_uuid}] Refining with Claude...")
            formatted_txt = temp_path / f"{base_name}_formatted.txt"
            if formatted_txt.exists():
                with open(formatted_txt, 'r', encoding='utf-8') as f:
                    formatted_text = f.read()

                refine_result = refine_transcript_text(
                    transcript_text=formatted_text,
                    output_path=formatted_txt,
                    verbose=False
                )

                # Step 5: Apply corrections to SRT
                logger.info(f"[{asset_uuid}] Applying SRT corrections...")
                changes_path = refine_result['changes_path']
                srt_path = format_result['srt_path']
                apply_corrections_to_srt_file(changes_path, srt_path)

            # Step 6: Post-process (embeddings, Comprehend, DB, S3)
            logger.info(f"[{asset_uuid}] Post-processing...")
            post_result = post_process_transcript(asset_uuid, str(output_base))

            result['status'] = 'success'
            result['embeddings'] = post_result.get('embeddings_count', 0)
            logger.info(f"[{asset_uuid}] [OK] Complete")

            # Clear CUDA cache after each asset
            torch.cuda.empty_cache()

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"[{asset_uuid}] FAILED: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Cleanup temp directory
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)

        return result


def worker_process(gpu_id: int, model_name: str, task_queue: Queue, result_queue: Queue):
    """
    Worker process entry point.

    Initializes worker, then loops pulling tasks from queue until STOP signal.

    Args:
        gpu_id: GPU device ID
        model_name: Whisper model size
        task_queue: Queue to receive tasks from
        result_queue: Queue to send results to
    """
    worker = TranscriptionWorker(gpu_id=gpu_id, model_name=model_name)

    try:
        worker.initialize()
        result_queue.put({'type': 'ready', 'gpu_id': gpu_id})

        while True:
            task = task_queue.get()

            if task == 'STOP':
                logger.info(f"Worker {gpu_id} received STOP signal")
                break

            asset_uuid = task['uuid']
            asset_name = task.get('name', 'Unknown')
            temp_dir = task['temp_dir']

            logger.info(f"Worker {gpu_id} processing: {asset_name}")

            result = worker.process_asset(asset_uuid, asset_name, temp_dir)
            result_queue.put({'type': 'result', 'data': result})

    except Exception as e:
        logger.error(f"Worker {gpu_id} crashed: {e}")
        result_queue.put({'type': 'error', 'gpu_id': gpu_id, 'error': str(e)})


class WorkerPool:
    """
    Manages pool of persistent transcription workers.
    """

    def __init__(self, gpu_ids: list, model_name: str = "medium"):
        """
        Initialize worker pool.

        Args:
            gpu_ids: List of GPU IDs to use (e.g., [0, 1])
            model_name: Whisper model size
        """
        self.gpu_ids = gpu_ids
        self.model_name = model_name
        self.workers = []
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.temp_base = Path("tmp")

    def start(self):
        """Start all worker processes."""
        logger.info(f"Starting {len(self.gpu_ids)} workers...")

        for gpu_id in self.gpu_ids:
            p = Process(
                target=worker_process,
                args=(gpu_id, self.model_name, self.task_queue, self.result_queue),
                name=f"Worker-GPU{gpu_id}"
            )
            p.start()
            self.workers.append(p)

        # Wait for all workers to initialize
        ready_count = 0
        while ready_count < len(self.gpu_ids):
            msg = self.result_queue.get()
            if msg['type'] == 'ready':
                ready_count += 1
                logger.info(f"Worker on GPU {msg['gpu_id']} ready ({ready_count}/{len(self.gpu_ids)})")

        logger.info(f"[OK] All {len(self.gpu_ids)} workers ready")

    def submit(self, asset_uuid: str, asset_name: str):
        """Submit an asset for processing."""
        temp_dir = self.temp_base / asset_uuid
        self.task_queue.put({
            'uuid': asset_uuid,
            'name': asset_name,
            'temp_dir': str(temp_dir)
        })

    def get_result(self, timeout: Optional[float] = None) -> Optional[Dict]:
        """Get next result from workers."""
        try:
            msg = self.result_queue.get(timeout=timeout)
            if msg['type'] == 'result':
                return msg['data']
            elif msg['type'] == 'error':
                logger.error(f"Worker error: {msg}")
                return None
        except:
            return None

    def shutdown(self):
        """Gracefully shutdown all workers."""
        logger.info("Shutting down workers...")

        # Send STOP to each worker
        for _ in self.workers:
            self.task_queue.put('STOP')

        # Wait for workers to finish
        for p in self.workers:
            p.join(timeout=30)
            if p.is_alive():
                logger.warning(f"Force terminating {p.name}")
                p.terminate()

        logger.info("[OK] All workers stopped")


if __name__ == '__main__':
    # Quick test
    print("Worker pool module loaded successfully")
    print("Run batch_process_pool.py to use the worker pool")
