#!/usr/bin/env python3
"""
Test speaker diarization to diagnose issues.
"""

import os
import sys
from pathlib import Path
import warnings

# Suppress PyTorch warnings (informational only, don't affect functionality)
warnings.filterwarnings('ignore', message='.*TF32.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*degrees of freedom.*', category=UserWarning)

# Fix torchaudio compatibility issue with pyannote.audio 4.0.1
import torchaudio

if not hasattr(torchaudio, 'AudioMetaData'):
    # Create a simple AudioMetaData class for compatibility
    from dataclasses import dataclass
    @dataclass
    class AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
    torchaudio.AudioMetaData = AudioMetaData

if not hasattr(torchaudio, 'list_audio_backends'):
    # Add missing list_audio_backends function
    def list_audio_backends():
        return ['soundfile']
    torchaudio.list_audio_backends = list_audio_backends

if not hasattr(torchaudio, 'info'):
    # Add missing info function for torchaudio 2.9
    def torchaudio_info(filepath, backend=None):
        import soundfile as sf
        info = sf.info(filepath)
        # Return AudioMetaData-like object
        return torchaudio.AudioMetaData(
            sample_rate=info.samplerate,
            num_frames=info.frames,
            num_channels=info.channels
        )
    torchaudio.info = torchaudio_info

from pyannote.audio import Pipeline

# Monkey-patch torchaudio.load to use soundfile instead of torchcodec
original_torchaudio_load = torchaudio.load

def patched_torchaudio_load(filepath, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None, backend=None):
    """Patched version that loads audio using soundfile instead of torchcodec"""
    import soundfile as sf
    import numpy as np
    import torch
    
    # Load with soundfile
    waveform_np, sample_rate = sf.read(filepath, dtype='float32', always_2d=True, start=frame_offset, frames=num_frames if num_frames > 0 else -1)
    
    # Convert to torch tensor and transpose to (channels, samples)
    waveform = torch.from_numpy(waveform_np.T if channels_first else waveform_np)
    
    return waveform, sample_rate

torchaudio.load = patched_torchaudio_load

# Monkey-patch pyannote's Audio class to use soundfile backend
from pyannote.audio.core.io import Audio
original_call = Audio.__call__

def patched_call(self, file):
    """Patched version that loads audio using soundfile instead of torchcodec"""
    import soundfile as sf
    import numpy as np
    import torch
    
    # Get the audio path from the file dict
    audio_path = file if isinstance(file, str) else file.get("audio", file)
    
    # Load with soundfile
    waveform_np, sample_rate = sf.read(audio_path, dtype='float32', always_2d=True)
    
    # Convert to torch tensor and transpose to (channels, samples)
    waveform = torch.from_numpy(waveform_np.T)
    
    return waveform, sample_rate

Audio.__call__ = patched_call
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
            use_auth_token=hf_token
        )
        print("   ✓ Pipeline loaded successfully")
        
        print("\n2. Running diarization...")
        print("   This may take several minutes for long audio files...")
        
        # Progress callback - accepts both positional and keyword arguments
        def progress_hook(*args, **kwargs):
            completed = kwargs.get('completed')
            total = kwargs.get('total')
            if completed is not None and total is not None and total > 0:
                progress = completed / total
                print(f"\r   Progress: {progress:.1%} ({completed}/{total})", end='', flush=True)
        
        diarization = pipeline(
            str(audio_path),
            min_speakers=2,
            max_speakers=10,
            hook=progress_hook
        )
        print()  # New line after progress
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Test speaker diarization')
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--hf-token', help='HuggingFace token (or set HF_TOKEN env var)')
    parser.add_argument('--diarization', action='store_true', default=False,
                        help='Enable speaker diarization (default: OFF)')
    
    args = parser.parse_args()
    
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    if args.diarization:
        test_diarization(args.audio_file, hf_token)
    else:
        print("Diarization is disabled. Use --diarization flag to enable.")
        print(f"Audio file: {args.audio_file}")
        print("\nTo enable diarization, run:")
        print(f"  python test_diarization.py \"{args.audio_file}\" --diarization")
