#!/usr/bin/env python3
"""
Master Transcription Pipeline

Orchestrates the complete workflow:
1. Transcribe audio with Whisper (vocabulary-enhanced with 32k token prompts)
2. Convert JSON to formatted text + SRT subtitles
3. Claude TXT refinement: fixes errors + adds proper paragraph breaks (single API call)
4. Apply corrections to SRT: reuses corrections from step 3 (no additional API call)

Usage:
    python transcribe_pipeline.py <audio_file> [options]

Example:
    python transcribe_pipeline.py "downloads/Clip 1_1969.mp3" --model medium
    python transcribe_pipeline.py audio.wav --model large --verbose
    python transcribe_pipeline.py audio.mp3 --skip-whisper --skip-claude
"""

import argparse
import sys
import subprocess
from pathlib import Path
import json


class TranscriptionPipeline:
    def __init__(self, audio_file, output_dir=None, model="medium", 
                 skip_whisper=False, skip_claude=False, verbose=True, device=None):
        self.audio_file = Path(audio_file)
        self.model = model
        self.skip_whisper = skip_whisper
        self.skip_claude = skip_claude
        self.verbose = verbose
        self.device = device
        
        # Get the directory where this script is located
        self.script_dir = Path(__file__).parent
        
        # Determine output directory
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path("downloads")
        
        # File paths for pipeline stages
        self.base_name = self.audio_file.stem
        self.json_file = self.output_dir / f"{self.base_name}.json"
        self.temp_srt_file = self.output_dir / f"{self.base_name}_temp.srt"  # intermediate
        self.formatted_file = self.output_dir / f"{self.base_name}_formatted.txt"  # intermediate
        self.refined_file = self.output_dir / f"{self.base_name}.txt"  # final
        self.logs_file = self.output_dir / f"{self.base_name}_logs.json"  # final (renamed from changes)
        self.srt_file = self.output_dir / f"{self.base_name}.srt"  # final (refined SRT takes this name)
    
    def run_command(self, cmd, stage_name):
        """Run a subprocess command and handle errors"""
        print(f"\n{'='*70}")
        print(f"STAGE: {stage_name}")
        print(f"{'='*70}")
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=not self.verbose,
                text=True
            )
            
            if not self.verbose and result.stdout:
                print(result.stdout)
            
            print(f"\n[OK] {stage_name} completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] {stage_name} failed!")
            if e.stderr:
                print(f"Error: {e.stderr}")
            return False
    
    def stage1_transcribe(self):
        """Stage 1: Transcribe with Whisper (vocabulary-enhanced)"""
        if self.skip_whisper:
            print(f"\n[SKIP]  Skipping Whisper transcription (--skip-whisper)")
            if not self.json_file.exists():
                print(f"[ERROR] Error: JSON file not found: {self.json_file}")
                return False
            return True
        
        if not self.audio_file.exists():
            print(f"[ERROR] Error: Audio file not found: {self.audio_file}")
            return False
        
        # Build command with absolute path to whisper_with_vocab.py
        whisper_script = self.script_dir / "whisper_with_vocab.py"
        cmd = [
            sys.executable,
            str(whisper_script),
            str(self.audio_file),
            "--model", self.model,
            "--output", str(self.output_dir / self.base_name)
        ]
        
        # Add device if specified
        if self.device:
            cmd.extend(["--device", self.device])
        
        return self.run_command(cmd, "Whisper Transcription (Vocabulary-Enhanced)")
    
    def stage2_format(self):
        """Stage 2: Convert to formatted text with paragraphs"""
        if not self.json_file.exists():
            print(f"[ERROR] Error: JSON file not found: {self.json_file}")
            return False
        
        # Build command with absolute path to convert_aws_transcribe.py
        convert_script = self.script_dir / "convert_aws_transcribe.py"
        cmd = [
            sys.executable,
            str(convert_script),
            str(self.json_file),
            "--output-dir", str(self.output_dir)
        ]
        
        return self.run_command(cmd, "Format with Intelligent Paragraphs")
    
    def stage3_refine(self):
        """Stage 3: Claude refinement for formatted transcript"""
        if self.skip_claude:
            print(f"\n[SKIP]  Skipping Claude refinement (--skip-claude)")
            return True
        
        if not self.formatted_file.exists():
            print(f"[ERROR] Error: Formatted file not found: {self.formatted_file}")
            return False
        
        # Build command for transcript refinement with absolute path
        claude_script = self.script_dir / "claude_refine_transcript.py"
        cmd = [
            sys.executable,
            str(claude_script),
            str(self.formatted_file)
        ]
        
        # Pass silent flag if disabled (verbose is default)
        if not self.verbose:
            cmd.append("--silent")
        
        return self.run_command(cmd, "Claude Transcript Refinement")
    
    def stage4_refine_srt(self):
        """Stage 4: Apply TXT corrections to SRT (no additional Claude call)"""
        if self.skip_claude:
            print(f"\n[SKIP]  Skipping SRT refinement (--skip-claude)")
            return True

        if not self.temp_srt_file.exists():
            print(f"[ERROR] Error: SRT file not found: {self.temp_srt_file}")
            return False

        # Check for logs file from Stage 3
        if not self.logs_file.exists():
            print(f"[ERROR] Error: Logs file not found: {self.logs_file}")
            print("       Stage 3 must complete successfully first.")
            return False

        # Build command to apply TXT corrections to SRT with absolute path
        apply_corrections_script = self.script_dir / "apply_txt_corrections_to_srt.py"
        cmd = [
            sys.executable,
            str(apply_corrections_script),
            str(self.logs_file),
            str(self.temp_srt_file)
        ]

        if self.verbose:
            cmd.append("--verbose")

        return self.run_command(cmd, "Apply TXT Corrections to SRT")
    
    def run(self):
        """Run the complete pipeline"""
        print(f"\n{'='*70}")
        print(f"TRANSCRIPTION PIPELINE")
        print(f"{'='*70}")
        print(f"Audio file: {self.audio_file}")
        print(f"Model: {self.model}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*70}\n")

        # Stage 1: Transcribe
        print("[STEP] Transcribing audio with Whisper...")
        if not self.stage1_transcribe():
            print("\n[ERROR] Pipeline failed at Stage 1 (Whisper)")
            return False

        # Stage 2: Format
        print("[STEP] Formatting transcript with intelligent paragraphs...")
        if not self.stage2_format():
            print("\n[ERROR] Pipeline failed at Stage 2 (Formatting)")
            return False

        # Stage 3: Claude Transcript Refinement
        print("[STEP] Refining transcript with Claude...")
        if not self.stage3_refine():
            print("\n[ERROR] Pipeline failed at Stage 3 (Claude Transcript)")
            return False

        # Stage 4: Claude SRT Refinement
        print("[STEP] Post-processing SRT corrections...")
        if not self.stage4_refine_srt():
            print("\n[ERROR] Pipeline failed at Stage 4 (Claude SRT)")
            return False

        # Cleanup intermediate files after successful completion
        if not self.skip_claude:
            for intermediate_file in [self.formatted_file, self.temp_srt_file]:
                if intermediate_file.exists():
                    intermediate_file.unlink()

        # Success!
        print(f"\n{'='*70}")
        print(f"[OK] Pipeline complete - transcription successful!")
        print(f"{'='*70}")
        print(f"\nOutput files:")
        print(f"  1. Raw transcript (JSON): {self.json_file}")
        if self.skip_claude:
            print(f"  2. Formatted text:        {self.formatted_file}")
            print(f"  3. Subtitles (SRT):       {self.temp_srt_file}")
        else:
            print(f"  2. Refined text:          {self.refined_file}")
            print(f"  3. Refined subtitles:     {self.srt_file}")
            print(f"  4. Change log:            {self.logs_file}")
        print(f"{'='*70}\n")

        return True


def main():
    parser = argparse.ArgumentParser(
        description='Complete transcription pipeline: Whisper → Format → Claude TXT → Apply to SRT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Stages:
  1. Whisper Transcription (vocabulary-enhanced with custom terms)
  2. Format with SRT generation (creates formatted text + subtitles)
  3. Claude TXT Refinement (fixes errors + adds proper paragraph breaks - one API call)
  4. Apply to SRT (reuses corrections from step 3 - no additional API call)

Examples:
  # Full pipeline with medium model
  python transcribe_pipeline.py "downloads/Clip 1_1969.mp3" --model medium
  
  # Full pipeline with large model (best quality)
  python transcribe_pipeline.py audio.wav --model large
  
  # Skip Whisper (use existing JSON)
  python transcribe_pipeline.py audio.mp3 --skip-whisper
  
  # Skip Claude (just transcribe and format)
  python transcribe_pipeline.py audio.mp3 --skip-claude
  
  # Silent mode (minimal output, no Claude thinking)
  python transcribe_pipeline.py audio.mp3 --model medium --silent
  
  # Custom output directory
  python transcribe_pipeline.py audio.mp3 --output-dir output/
        """
    )
    
    parser.add_argument('audio_file', 
                        help='Path to audio file (mp3, wav, m4a, etc.)')
    
    parser.add_argument('-m', '--model', 
                        default='medium',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size (default: medium)')
    
    parser.add_argument('-o', '--output-dir',
                        help='Output directory (default: downloads/)')
    
    parser.add_argument('--skip-whisper', 
                        action='store_true',
                        help='Skip Whisper transcription (use existing JSON)')
    
    parser.add_argument('--skip-claude', 
                        action='store_true',
                        help='Skip Claude refinement (faster, less accurate)')
    
    parser.add_argument('-d', '--device',
                        help='CUDA device to use (e.g., cuda:0, cuda:1)')
    
    parser.add_argument('-s', '--silent', 
                        action='store_true',
                        help='Silent mode - disable detailed output and Claude thinking display')
    
    args = parser.parse_args()
    
    # Validate audio file
    audio_path = Path(args.audio_file)
    if not args.skip_whisper and not audio_path.exists():
        print(f"[ERROR] Error: Audio file not found: {args.audio_file}")
        return 1
    
    # Verbose is default (ON unless --silent is specified)
    verbose = not args.silent
    
    # Run pipeline
    pipeline = TranscriptionPipeline(
        audio_file=args.audio_file,
        output_dir=args.output_dir,
        model=args.model,
        skip_whisper=args.skip_whisper,
        skip_claude=args.skip_claude,
        verbose=verbose,
        device=args.device
    )
    
    success = pipeline.run()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
