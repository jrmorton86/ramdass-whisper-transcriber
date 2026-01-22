"""
Resource Manager - Tracks GPU slots and system RAM for job scheduling.
"""

import asyncio
import logging
import subprocess
import time
from typing import Optional

import psutil

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Manages GPU slot allocation and system RAM monitoring.

    Tracks available slots per GPU based on VRAM capacity.
    Ensures system RAM threshold is met before starting jobs.
    """

    # Resource thresholds
    VRAM_PER_JOB_MB = 3000      # ~3GB for Whisper medium model
    VRAM_HEADROOM_MB = 2000     # 2GB safety buffer per GPU
    MIN_SYSTEM_RAM_MB = 4000    # Require 4GB free before starting job

    def __init__(self, gpu_ids: list[int]):
        """
        Initialize resource manager.

        Args:
            gpu_ids: List of GPU IDs to manage (from settings.gpu_ids)
        """
        self._gpu_ids = gpu_ids
        self._gpu_info: dict[int, dict] = {}  # {gpu_id: {name, total_mb, max_slots, jobs: set}}
        self._lock = asyncio.Lock()
        self._slot_available = asyncio.Event()
        self._initialized = False

    async def initialize(self):
        """Detect GPUs and calculate available slots."""
        try:
            self._gpu_info = self._detect_gpus()
            self._initialized = True

            total_slots = sum(info["max_slots"] for info in self._gpu_info.values())
            gpu_summary = ", ".join(
                f"GPU {gpu_id} ({info['name']}): {info['max_slots']} slots"
                for gpu_id, info in self._gpu_info.items()
            )
            logger.info(f"Resource manager initialized: {total_slots} total slots")
            logger.info(f"  {gpu_summary}")

            # Set event since slots are available
            self._slot_available.set()

        except Exception as e:
            logger.warning(f"GPU detection failed: {e}, falling back to single GPU with 1 slot")
            self._gpu_info = {
                self._gpu_ids[0] if self._gpu_ids else 0: {
                    "name": "Unknown",
                    "total_mb": 8000,
                    "max_slots": 1,
                    "jobs": set(),
                }
            }
            self._initialized = True
            self._slot_available.set()

    def _detect_gpus(self) -> dict[int, dict]:
        """
        Query GPU information using nvidia-smi.

        Returns:
            Dict mapping gpu_id to {name, total_mb, max_slots, jobs}
        """
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {result.stderr}")

        gpu_info = {}
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpu_id = int(parts[0])

                # Only track GPUs we're configured to use
                if gpu_id not in self._gpu_ids:
                    continue

                name = parts[1]
                total_mb = int(parts[2])

                # Calculate max slots: (total - headroom) / per_job
                available_mb = total_mb - self.VRAM_HEADROOM_MB
                max_slots = max(1, available_mb // self.VRAM_PER_JOB_MB)

                gpu_info[gpu_id] = {
                    "name": name,
                    "total_mb": total_mb,
                    "max_slots": max_slots,
                    "jobs": set(),
                }

                logger.info(f"GPU {gpu_id} ({name}): {total_mb} MB VRAM, {max_slots} slots available")

        if not gpu_info:
            raise RuntimeError("No configured GPUs found")

        return gpu_info

    def _find_available_gpu(self) -> Optional[int]:
        """
        Find a GPU with an available slot.

        Returns:
            GPU ID with available slot, or None if all full
        """
        for gpu_id, info in self._gpu_info.items():
            if len(info["jobs"]) < info["max_slots"]:
                return gpu_id
        return None

    async def wait_for_slot(self) -> int:
        """
        Wait for an available GPU slot.

        Returns:
            GPU ID that has an available slot
        """
        while True:
            async with self._lock:
                gpu_id = self._find_available_gpu()
                if gpu_id is not None:
                    return gpu_id

                # Clear event so we wait
                self._slot_available.clear()

            # Wait for a slot to become available
            await self._slot_available.wait()

    async def reserve_slot(self, gpu_id: int, job_id: str) -> bool:
        """
        Reserve a slot on a GPU for a job.

        Args:
            gpu_id: GPU to reserve slot on
            job_id: Job ID reserving the slot

        Returns:
            True if reserved successfully, False if no slot available
        """
        async with self._lock:
            if gpu_id not in self._gpu_info:
                logger.error(f"Unknown GPU {gpu_id}")
                return False

            info = self._gpu_info[gpu_id]
            if len(info["jobs"]) >= info["max_slots"]:
                logger.warning(f"GPU {gpu_id} has no available slots")
                return False

            info["jobs"].add(job_id)
            slots_used = len(info["jobs"])
            slots_max = info["max_slots"]

            logger.info(f"Job {job_id} assigned to GPU {gpu_id} (slot {slots_used}/{slots_max} now used)")
            return True

    async def release_slot(self, gpu_id: int, job_id: str):
        """
        Release a slot when a job completes.

        Args:
            gpu_id: GPU to release slot on
            job_id: Job ID releasing the slot
        """
        async with self._lock:
            if gpu_id not in self._gpu_info:
                return

            info = self._gpu_info[gpu_id]
            info["jobs"].discard(job_id)

            slots_used = len(info["jobs"])
            slots_max = info["max_slots"]

            logger.info(f"Job {job_id} completed, GPU {gpu_id} slot released ({slots_used}/{slots_max} now used)")

            # Signal that a slot is available
            self._slot_available.set()

    def check_system_ram(self) -> bool:
        """
        Check if enough system RAM is available.

        Returns:
            True if RAM >= MIN_SYSTEM_RAM_MB, False otherwise
        """
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        return available_mb >= self.MIN_SYSTEM_RAM_MB

    async def wait_for_ram(self, timeout: float = 300) -> bool:
        """
        Wait for system RAM to become available.

        Args:
            timeout: Maximum time in seconds to wait for RAM (default: 300)

        Returns:
            True if RAM became available, False if timeout
        """
        start_time = time.time()

        while not self.check_system_ram():
            elapsed = time.time() - start_time
            if elapsed > timeout:
                available_mb = psutil.virtual_memory().available / (1024 * 1024)
                logger.warning(
                    f"RAM wait timeout after {timeout}s, "
                    f"proceeding with {available_mb:.0f} MB available"
                )
                return False

            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            logger.info(f"System RAM low ({available_mb:.0f} MB free), waiting...")
            await asyncio.sleep(5)

        return True

    def get_status(self) -> dict:
        """
        Get current resource status for API/dashboard.

        Returns:
            Dict with GPU and RAM status
        """
        mem = psutil.virtual_memory()

        gpus = []
        for gpu_id, info in self._gpu_info.items():
            gpus.append({
                "id": gpu_id,
                "name": info["name"],
                "total_mb": info["total_mb"],
                "slots_used": len(info["jobs"]),
                "slots_max": info["max_slots"],
                "jobs": list(info["jobs"]),
            })

        return {
            "gpus": gpus,
            "system_ram_mb": int(mem.total / (1024 * 1024)),
            "system_ram_available_mb": int(mem.available / (1024 * 1024)),
            "system_ram_threshold_mb": self.MIN_SYSTEM_RAM_MB,
        }
