#!/usr/bin/env python3
"""
GPU Memory Monitor - Check VRAM usage during transcription

Run this script to verify that Whisper is using GPU VRAM instead of system RAM.
"""

import subprocess
import sys
from pathlib import Path

def check_cuda_available():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA is available")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   PyTorch Version: {torch.__version__}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"           Total VRAM: {props.total_memory / 1024**3:.2f} GB")
            return True
        else:
            print("❌ CUDA is not available")
            print("   Whisper will use CPU and system RAM")
            return False
    except ImportError:
        print("❌ PyTorch is not installed")
        return False

def get_gpu_memory_usage():
    """Get current GPU memory usage via nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("\n📊 Current GPU Memory Usage:")
            print("="*80)
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpu_id, name, mem_used, mem_total, util = parts
                    mem_used_gb = float(mem_used) / 1024
                    mem_total_gb = float(mem_total) / 1024
                    mem_percent = (float(mem_used) / float(mem_total)) * 100
                    print(f"GPU {gpu_id}: {name}")
                    print(f"  VRAM: {mem_used_gb:.2f} / {mem_total_gb:.2f} GB ({mem_percent:.1f}%)")
                    print(f"  Utilization: {util}%")
                    print()
        else:
            print("❌ Failed to query nvidia-smi")
            
    except Exception as e:
        print(f"❌ Error querying GPU memory: {e}")

def test_whisper_gpu_usage():
    """Test that Whisper loads to GPU"""
    print("\n🧪 Testing Whisper GPU Loading...")
    print("="*80)
    
    try:
        import torch
        import whisper
        
        print("Loading Whisper 'tiny' model to test GPU usage...")
        
        # Test auto-detection (should use CUDA if available)
        if torch.cuda.is_available():
            print("\n1️⃣ Testing auto-detection (should use CUDA):")
            model = whisper.load_model("tiny")
            device = next(model.parameters()).device
            print(f"   Model loaded on: {device}")
            if device.type == 'cuda':
                print("   ✅ SUCCESS: Model is on GPU")
            else:
                print("   ⚠️  WARNING: Model is on CPU")
            del model
            torch.cuda.empty_cache()
            
            print("\n2️⃣ Testing explicit CUDA device:")
            model = whisper.load_model("tiny", device="cuda")
            device = next(model.parameters()).device
            print(f"   Model loaded on: {device}")
            if device.type == 'cuda':
                print("   ✅ SUCCESS: Model is on GPU")
            else:
                print("   ❌ FAILED: Model is on CPU")
            del model
            torch.cuda.empty_cache()
            
            print("\n3️⃣ Checking FP16 support:")
            model = whisper.load_model("tiny", device="cuda")
            # Check if model parameters are in FP16
            dtype = next(model.parameters()).dtype
            print(f"   Model dtype: {dtype}")
            if dtype == torch.float16:
                print("   ✅ Model is in FP16 (optimal VRAM usage)")
            else:
                print(f"   ℹ️  Model is in {dtype} (FP16 applied during transcription)")
            del model
            torch.cuda.empty_cache()
        else:
            print("❌ CUDA not available - cannot test GPU loading")
            
    except ImportError as e:
        print(f"❌ Required package not installed: {e}")
    except Exception as e:
        print(f"❌ Error during test: {e}")

def main():
    print("="*80)
    print("GPU MEMORY MONITOR - Whisper VRAM Usage Verification")
    print("="*80)
    print()
    
    # Check CUDA availability
    cuda_available = check_cuda_available()
    
    if cuda_available:
        # Show current GPU memory usage
        get_gpu_memory_usage()
        
        # Test Whisper GPU loading
        response = input("\nTest Whisper GPU loading? This will load a tiny model. (y/n): ").strip().lower()
        if response == 'y':
            test_whisper_gpu_usage()
            print("\n📊 GPU Memory After Test:")
            get_gpu_memory_usage()
    
    print("\n" + "="*80)
    print("💡 Tips for optimal VRAM usage:")
    print("="*80)
    print("1. Models load directly to GPU when CUDA is available")
    print("2. FP16 is auto-enabled on CUDA (50% less VRAM)")
    print("3. Use --device cuda:0 or cuda:1 to target specific GPU")
    print("4. Use --experimental mode for automatic GPU balancing")
    print("5. Monitor with: nvidia-smi -l 1 (updates every second)")
    print("="*80)

if __name__ == '__main__':
    main()
