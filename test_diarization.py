#!/usr/bin/env python3
"""
Test speaker diarization to diagnose issues.
"""

import os
import sys
from pathlib import Path
from pyannote.audio import Pipeline
import torch

def test_diarization(audio_path, hf_token):
    """Test if diarization works properly"""
    
    print("="*60)
    print("DIARIZATION TEST")
    print("="*60)
    
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    print(f"\nAudio file: {audio_path}")
    print(f"HF Token: {hf_token[:10]}..." if hf_token else "No token provided")
    
    if not hf_token:
        print("\nERROR: No HuggingFace token provided!")
        return
    
    try:
        print("\n1. Loading pyannote pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        print("   ✓ Pipeline loaded successfully")
        
        print("\n2. Running diarization...")
        print("   This may take several minutes for long audio files...")
        
        diarization = pipeline(
            str(audio_path),
            min_speakers=2,
            max_speakers=10
        )
        print("   ✓ Diarization complete!")
        
        print("\n3. Results:")
        print("-"*60)
        
        # Count speakers
        speakers = set()
        segment_count = 0
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            segment_count += 1
            
            # Print first 10 segments
            if segment_count <= 10:
                print(f"   [{turn.start:06.2f}s - {turn.end:06.2f}s] {speaker}")
        
        print(f"\n   Total segments: {segment_count}")
        print(f"   Unique speakers: {len(speakers)}")
        print(f"   Speaker labels: {', '.join(sorted(speakers))}")
        
        print("\n" + "="*60)
        print("✓ TEST SUCCESSFUL")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60)
        print("✗ TEST FAILED")
        print("="*60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_diarization.py <audio_file>")
        print("\nSet HF_TOKEN environment variable or provide --hf-token")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token and len(sys.argv) > 2:
        hf_token = sys.argv[2]
    
    test_diarization(audio_file, hf_token)
