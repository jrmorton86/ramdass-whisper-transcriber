#!/usr/bin/env python3
"""
Transcribe with diarization using a fallback approach for Windows compatibility.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import whisper
import torch
import librosa

# Manually import what we need from pyannote
try:
    from pyannote.audio import Pipeline
except Exception as e:
    print(f"Warning: Could not import pyannote.audio: {e}")

def test_simple_diarization(audio_path, hf_token):
    """Test speaker diarization with error handling"""
    
    print("="*60)
    print("SIMPLE DIARIZATION TEST")
    print("="*60)
    
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    print(f"\nAudio file: {audio_path}")
    
    # Load audio with librosa for compatibility
    print("\nLoading audio with librosa...")
    try:
        y, sr = librosa.load(str(audio_path), sr=None)
        print(f"✓ Audio loaded: {len(y)/sr:.2f}s at {sr}Hz")
    except Exception as e:
        print(f"Error loading audio: {e}")
        return
    
    # Try to load pipeline
    if not hf_token:
        print("No HuggingFace token provided - skipping diarization")
        return
    
    print("\nLoading diarization pipeline...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        print("✓ Pipeline loaded successfully")
        
        # Create a dict format that pyannote expects
        audio_dict = {
            "waveform": torch.from_numpy(y).unsqueeze(0).float(),
            "sample_rate": sr
        }
        
        print("\nRunning diarization...")
        diarization = pipeline(audio_dict, min_speakers=2, max_speakers=10)
        
        print("✓ Diarization complete!")
        
        # Extract results
        speakers = set()
        segment_count = 0
        
        print("\nResults:")
        print("-"*60)
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            segment_count += 1
            
            if segment_count <= 10:
                print(f"  [{turn.start:06.2f}s - {turn.end:06.2f}s] {speaker}")
        
        print(f"\n✓ Total segments: {segment_count}")
        print(f"✓ Unique speakers: {len(speakers)}")
        print(f"✓ Speaker labels: {', '.join(sorted(speakers))}")
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_diarization.py <audio_file>")
        sys.exit(1)
    
    import sys
    audio_file = sys.argv[1]
    hf_token = os.environ.get("HF_TOKEN")
    
    test_simple_diarization(audio_file, hf_token)
