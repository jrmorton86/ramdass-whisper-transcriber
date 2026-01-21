#!/usr/bin/env python3
"""
GPU Load Balancer - Dynamically assign tasks to least-loaded GPU

Monitors GPU utilization and assigns work to the GPU with lowest load.
Uses nvidia-smi to query GPU metrics and tracks active tasks per device.
"""

import subprocess
import threading
import logging
from typing import Dict, Optional
import time

logger = logging.getLogger(__name__)


class GPULoadBalancer:
    """
    Intelligent GPU load balancer for multi-GPU systems.
    
    Monitors GPU utilization and assigns tasks to the least loaded GPU.
    Tracks active tasks per GPU to prevent overloading.
    """
    
    def __init__(self, gpu_ids: list[int] = [0, 1], max_tasks_per_gpu: int = 5):
        """
        Initialize GPU load balancer.
        
        Args:
            gpu_ids: List of GPU device IDs to balance across (e.g., [0, 1])
            max_tasks_per_gpu: Maximum concurrent tasks per GPU (default: 5)
        """
        self.gpu_ids = gpu_ids
        self.max_tasks_per_gpu = max_tasks_per_gpu
        self.lock = threading.Lock()
        
        # Track active tasks per GPU
        self.active_tasks: Dict[int, int] = {gpu_id: 0 for gpu_id in gpu_ids}
        
        logger.info(f"🎮 GPU Load Balancer initialized")
        logger.info(f"   GPUs: {gpu_ids}")
        logger.info(f"   Max tasks per GPU: {max_tasks_per_gpu}")
    
    def get_gpu_utilization(self) -> Dict[int, float]:
        """
        Query GPU utilization using nvidia-smi.
        
        Returns:
            Dict mapping GPU ID to utilization percentage (0-100)
        """
        try:
            # Query nvidia-smi for GPU utilization
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                logger.warning("Failed to query nvidia-smi, using task count only")
                return {gpu_id: 0.0 for gpu_id in self.gpu_ids}
            
            # Parse output: "0, 45\n1, 32\n"
            utilization = {}
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    gpu_id = int(parts[0].strip())
                    util = float(parts[1].strip())
                    if gpu_id in self.gpu_ids:
                        utilization[gpu_id] = util
            
            return utilization
            
        except Exception as e:
            logger.warning(f"Error querying GPU utilization: {e}")
            return {gpu_id: 0.0 for gpu_id in self.gpu_ids}
    
    def get_best_device(self) -> str:
        """
        Get the CUDA device with lowest current load.
        
        Considers both GPU utilization and active task count.
        
        Returns:
            Device string (e.g., 'cuda:0', 'cuda:1')
        """
        with self.lock:
            # Get current GPU utilization
            utilization = self.get_gpu_utilization()
            
            # Calculate load score for each GPU
            # Score = (active_tasks / max_tasks) * 50 + (utilization / 100) * 50
            # Lower score = better choice
            scores = {}
            for gpu_id in self.gpu_ids:
                task_load = (self.active_tasks[gpu_id] / self.max_tasks_per_gpu) * 50
                util_load = (utilization.get(gpu_id, 0) / 100) * 50
                scores[gpu_id] = task_load + util_load
            
            # Select GPU with lowest score
            best_gpu = min(scores.keys(), key=lambda k: scores[k])
            
            # Increment task count
            self.active_tasks[best_gpu] += 1
            
            logger.debug(f"GPU Selection - Active tasks: {self.active_tasks}, "
                        f"Utilization: {utilization}, Scores: {scores} -> GPU {best_gpu}")
            
            return f"cuda:{best_gpu}"
    
    def release_device(self, device: str):
        """
        Mark a task as complete and decrement the active task count.
        
        Args:
            device: Device string (e.g., 'cuda:0', 'cuda:1')
        """
        # Extract GPU ID from device string
        gpu_id = int(device.split(':')[1])
        
        with self.lock:
            if gpu_id in self.active_tasks and self.active_tasks[gpu_id] > 0:
                self.active_tasks[gpu_id] -= 1
                logger.debug(f"Released {device}, active tasks: {self.active_tasks}")
    
    def get_status(self) -> str:
        """
        Get current load balancer status as formatted string.
        
        Returns:
            Status string with GPU loads
        """
        with self.lock:
            utilization = self.get_gpu_utilization()
            status_lines = []
            for gpu_id in self.gpu_ids:
                tasks = self.active_tasks[gpu_id]
                util = utilization.get(gpu_id, 0)
                status_lines.append(
                    f"GPU {gpu_id}: {tasks}/{self.max_tasks_per_gpu} tasks, {util:.0f}% util"
                )
            return " | ".join(status_lines)


def test_load_balancer():
    """Test the GPU load balancer"""
    print("Testing GPU Load Balancer...")
    
    balancer = GPULoadBalancer(gpu_ids=[0, 1], max_tasks_per_gpu=5)
    
    print("\nGetting GPU utilization:")
    util = balancer.get_gpu_utilization()
    print(f"Utilization: {util}")
    
    print("\nSimulating task assignment:")
    devices = []
    for i in range(8):
        device = balancer.get_best_device()
        devices.append(device)
        print(f"Task {i+1}: {device} - Status: {balancer.get_status()}")
        time.sleep(0.1)
    
    print("\nReleasing tasks:")
    for i, device in enumerate(devices[:4]):
        balancer.release_device(device)
        print(f"Released task {i+1} ({device}) - Status: {balancer.get_status()}")
    
    print("\nTest complete!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_load_balancer()
