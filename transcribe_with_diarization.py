#!/usr/bin/env python3
"""
Transcribe audio files with speaker diarization.
Uses Whisper for transcription and pyannote.audio for speaker identification.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import whisper
from pyannote.audio import Pipeline
import torch
from speaker_config import SpeakerConfig
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def transcribe_with_speakers(audio_path, output_dir=None, model_size="base", hf_token=None, min_speakers=None, max_speakers=None, speaker_config=None, speaker_names=None):
    """
    Transcribe audio with speaker diarization and identification.
    
    Args:
        audio_path: Path to audio file
        output_dir: Directory to save output files (default: same as audio file)
        model_size: Whisper model size (tiny, base, small, medium, large)
        hf_token: Hugging Face token for pyannote.audio models
        min_speakers: Minimum number of speakers (default: 2)
        max_speakers: Maximum number of speakers (default: 10)
        speaker_config: SpeakerConfig object or path to config JSON
        speaker_names: List of names to auto-map to detected speakers (e.g., ["Ram Dass", "Host"])
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    # Set output directory
    if output_dir is None:
        output_dir = audio_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_base = output_dir / audio_path.stem
    
    # Initialize speaker config
    if speaker_config is None:
        speaker_config = SpeakerConfig()
    elif isinstance(speaker_config, str):
        speaker_config = SpeakerConfig(speaker_config)
    
    print(f"Processing: {audio_path.name}")
    print(f"Output will be saved to: {output_dir}")
    print("-" * 60)
    
    # Display speaker configuration if any
    if speaker_config.speakers or speaker_names:
        speaker_config.list_speakers()
    
    # Load Whisper model
    print(f"\n1. Loading Whisper model ({model_size})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    whisper_model = whisper.load_model(model_size, device=device)
    
    # Transcribe with Whisper
    print("\n2. Transcribing audio with Whisper...")
    print("   (This may take a few minutes depending on audio length...)")
    result = whisper_model.transcribe(
        str(audio_path),
        language="en",
        verbose=True  # Show transcription progress
    )
    
    print(f"\n   ✓ Detected language: {result['language']}")
    print(f"   ✓ Found {len(result['segments'])} segments")
    
    # Perform speaker diarization
    print("\n3. Performing speaker diarization...")
    if hf_token:
        try:
            print("   Loading diarization pipeline...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
            
            # Move pipeline to GPU if available for faster processing
            if torch.cuda.is_available():
                print("   Using GPU acceleration for diarization")
                pipeline.to(torch.device("cuda"))
            else:
                print("   Using CPU for diarization (GPU would be faster)")
            
            print("   Running speaker detection (this may take a few minutes)...")
            
            # Set up diarization parameters
            diarization_params = {}
            if min_speakers is not None:
                diarization_params["min_speakers"] = min_speakers
            else:
                diarization_params["min_speakers"] = 2
            if max_speakers is not None:
                diarization_params["max_speakers"] = max_speakers
            else:
                diarization_params["max_speakers"] = 10
            
            # Run diarization with progress monitoring
            try:
                from pyannote.audio.pipelines.utils.hook import ProgressHook
                with ProgressHook() as hook:
                    diarization = pipeline(str(audio_path), hook=hook, **diarization_params)
            except ImportError:
                # Fallback if ProgressHook is not available
                print("   (Install pyannote.audio>=3.0 for progress monitoring)")
                diarization = pipeline(str(audio_path), **diarization_params)
            
            # Extract speaker segments
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            unique_speakers = set([s['speaker'] for s in speaker_segments])
            print(f"   ✓ Found {len(unique_speakers)} unique speakers: {', '.join(sorted(unique_speakers))}")
            print(f"   ✓ Total speaker segments: {len(speaker_segments)}")
            
            # Auto-map speakers if names are provided
            if speaker_names and len(speaker_names) > 0:
                print(f"\n   Auto-mapping speakers to provided names:")
                speaker_config.auto_map_speakers(list(unique_speakers), speaker_names)
                for dia_speaker in sorted(unique_speakers):
                    mapped_name = speaker_config.get_speaker_name(dia_speaker)
                    if mapped_name != dia_speaker:
                        print(f"     {dia_speaker} -> {mapped_name}")
            
            # Merge transcription with speaker labels
            print("\n4. Merging transcription with speaker labels...")
            output_segments = []
            
            for segment in tqdm(result['segments'], desc="   Processing segments", unit="segment"):
                seg_start = segment['start']
                seg_end = segment['end']
                seg_text = segment['text'].strip()
                
                # Find which speaker was talking during this segment
                # Use the speaker who spoke the most during this time
                speaker_times = {}
                for sp_seg in speaker_segments:
                    overlap_start = max(seg_start, sp_seg['start'])
                    overlap_end = min(seg_end, sp_seg['end'])
                    overlap = max(0, overlap_end - overlap_start)
                    
                    if overlap > 0:
                        speaker = sp_seg['speaker']
                        speaker_times[speaker] = speaker_times.get(speaker, 0) + overlap
                
                # Assign to speaker with most overlap
                if speaker_times:
                    assigned_speaker = max(speaker_times, key=speaker_times.get)
                else:
                    assigned_speaker = "UNKNOWN"
                
                # Map speaker label to configured name
                speaker_name = speaker_config.get_speaker_name(assigned_speaker)
                
                output_segments.append({
                    "start": seg_start,
                    "end": seg_end,
                    "speaker": speaker_name,
                    "text": seg_text
                })
        
        except Exception as e:
            print(f"   Warning: Diarization failed: {e}")
            print("   Continuing with transcription only...")
            output_segments = [
                {
                    "start": seg['start'],
                    "end": seg['end'],
                    "speaker": speaker_config.get_speaker_name("SPEAKER_00"),
                    "text": seg['text'].strip()
                }
                for seg in result['segments']
            ]
    else:
        print("   Skipping diarization (no HuggingFace token provided)")
        print("   All segments will be labeled as SPEAKER_00")
        output_segments = [
            {
                "start": seg['start'],
                "end": seg['end'],
                "speaker": speaker_config.get_speaker_name("SPEAKER_00"),
                "text": seg['text'].strip()
            }
            for seg in result['segments']
        ]
    
    # Save outputs
    print(f"\n5. Saving outputs...")
    
    # Save JSON with detailed segments
    json_file = f"{output_base}_with_speakers.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "audio_file": str(audio_path),
            "segments": output_segments,
            "full_text": result['text']
        }, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved JSON: {json_file}")
    
    # Save formatted transcript
    txt_file = f"{output_base}_with_speakers.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        current_speaker = None
        for seg in output_segments:
            timestamp = f"[{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}]"
            
            # Add speaker label when speaker changes
            if seg['speaker'] != current_speaker:
                f.write(f"\n{seg['speaker']}:\n")
                current_speaker = seg['speaker']
            
            f.write(f"{timestamp} {seg['text']}\n")
    print(f"   ✓ Saved transcript: {txt_file}")
    
    # Save simple text without timestamps
    simple_txt_file = f"{output_base}_simple.txt"
    with open(simple_txt_file, 'w', encoding='utf-8') as f:
        current_speaker = None
        for seg in output_segments:
            if seg['speaker'] != current_speaker:
                if current_speaker is not None:
                    f.write("\n\n")
                f.write(f"{seg['speaker']}:\n")
                current_speaker = seg['speaker']
            f.write(f"{seg['text']}\n")
    print(f"   ✓ Saved simple text: {simple_txt_file}")
    
    print("\n" + "=" * 60)
    print("✓ Processing complete!")
    print("=" * 60)


def format_timestamp(seconds):
    """Format seconds as HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with speaker diarization using Whisper and pyannote.audio"
    )
    parser.add_argument(
        "audio_file",
        help="Path to audio file (mp3, wav, mp4, etc.)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory (default: same as audio file)"
    )
    parser.add_argument(
        "-m", "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--hf-token",
        help="HuggingFace token for pyannote.audio (required for diarization)"
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        help="Minimum number of speakers (default: 2)"
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="Maximum number of speakers (default: 10)"
    )
    parser.add_argument(
        "--speaker-names",
        type=str,
        help="Comma-separated list of speaker names (e.g., 'Ram Dass,Host,Caller')"
    )
    parser.add_argument(
        "--speaker-config",
        type=str,
        help="Path to speaker configuration JSON file"
    )
    
    args = parser.parse_args()
    
    # Check for HF token in environment if not provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    if not hf_token:
        print("\n" + "!" * 60)
        print("WARNING: No HuggingFace token provided!")
        print("Speaker diarization will be skipped.")
        print("\nTo enable diarization:")
        print("1. Get a token from https://huggingface.co/settings/tokens")
        print("2. Accept the model terms at:")
        print("   https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("3. Provide token via --hf-token or HF_TOKEN environment variable")
        print("!" * 60 + "\n")
        
        response = input("Continue without diarization? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Parse speaker names if provided
    speaker_names = None
    if args.speaker_names:
        speaker_names = [name.strip() for name in args.speaker_names.split(',')]
    
    transcribe_with_speakers(
        args.audio_file,
        args.output_dir,
        args.model,
        hf_token,
        args.min_speakers,
        args.max_speakers,
        args.speaker_config,
        speaker_names
    )


if __name__ == "__main__":
    main()
