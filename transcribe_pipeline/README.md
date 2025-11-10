# Ram Dass Whisper Transcriber

Production-ready transcription pipeline for Ram Dass audio content using OpenAI Whisper and Claude AI. Features vocabulary-enhanced transcription, intelligent paragraphing, and cost-optimized AI refinement.

## Features

- 🎙️ **Vocabulary-Enhanced Transcription**: 60+ custom spiritual/philosophical terms (Ouspensky, Maharaj-ji, Gurdjieff, etc.)
- 🤖 **AI-Powered Refinement**: Claude Sonnet 4.5 for intelligent error correction and paragraphing
- 📝 **Intelligent Paragraphs**: Q3-based pause detection for natural reading flow
- 💰 **Cost Optimized**: Single Claude API call (~$0.05 per file)
- � **SRT Subtitles**: Automatically generated and refined
- 📊 **GPU Acceleration**: CUDA support for fast Whisper processing
- 🚀 **Production Ready**: 4-stage pipeline with comprehensive error handling

## Prerequisites

- Python 3.12+
- FFmpeg (for audio processing)
- CUDA-capable GPU (recommended for Whisper)
- AWS account with Bedrock access (for Claude API)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jrmorton86/ramdass-whisper-transcriber.git
   cd ramdass-whisper-transcriber
   ```

2. **Create a virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure AWS Bedrock** (for Claude AI refinement):
   ```powershell
   # Configure AWS credentials
   aws configure
   # Access Key ID: YOUR_ACCESS_KEY
   # Secret Access Key: YOUR_SECRET_KEY
   # Default region: us-east-1
   
   # Ensure Bedrock access with Claude Sonnet 4.5 enabled
   ```

## Quick Start

### Basic Usage

```powershell
# Full pipeline (all 4 stages)
python transcribe_pipeline.py "downloads\Clip 1_1969.mp3" --model medium

# Skip Whisper (use existing JSON)
python transcribe_pipeline.py "downloads\Clip 1_1969.mp3" --skip-whisper

# Skip Claude refinement
python transcribe_pipeline.py "downloads\Clip 1_1969.mp3" --skip-claude

# Verbose output with detailed progress
python transcribe_pipeline.py "downloads\Clip 1_1969.mp3" --model large --verbose
```

### Options

- `-m, --model`: Whisper model size (tiny, base, small, medium, large)
  - Default: `medium`
  - **Recommended**: `medium` (best quality/speed balance)
  - `large` for production quality (slower, 10GB VRAM)

- `-o, --output-dir`: Output directory (default: `downloads/`)

- `--skip-whisper`: Skip transcription (use existing JSON)
  - Useful for re-running formatting/refinement only

- `--skip-claude`: Skip Claude AI refinement
  - Faster, but no intelligent corrections or paragraphing

- `-v, --verbose`: Show detailed progress, timing, and costs

## Output Files

The pipeline generates 8 files:

1. **`{filename}.json`**: Raw Whisper output (word-level timing)
2. **`{filename}.txt`**: Simple transcript
3. **`{filename}.srt`**: Original SRT subtitles
4. **`{filename}_formatted.txt`**: With intelligent paragraph breaks
5. **`{filename}_formatted_refined.txt`**: Final refined transcript ⭐
6. **`{filename}_formatted_refined_changes.json`**: List of corrections made
7. **`{filename}_refined.srt`**: Final refined subtitles ⭐
8. **`{filename}_refined_changes.json`**: SRT corrections log

## Pipeline Stages

```
Stage 1: Whisper Transcription (~4 min for 20-min audio)
  ↓ Vocabulary-enhanced with 60+ custom terms
  
Stage 2: Format & Generate SRT (~5 seconds)
  ↓ Q3-based intelligent paragraph breaks
  
Stage 3: Claude TXT Refinement (~45 seconds, ~$0.05)
  ↓ Fix errors + add proper paragraphs
  
Stage 4: Apply to SRT (~2 seconds, $0 - reuses Stage 3!)
  ✓ Final refined transcript + subtitles
```

## Example Workflows

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Single file production quality
python transcribe_pipeline.py "audio.mp3" --model large --verbose

# Batch process multiple files
Get-ChildItem .\audio\*.mp3 | ForEach-Object { 
    python transcribe_pipeline.py $_.FullName --model medium
}

# Re-run refinement only (after vocabulary update)
python transcribe_pipeline.py "audio.mp3" --skip-whisper
```

## Project Structure

```
ramdass-whisper-transcriber/
├── transcribe_pipeline.py              # Main orchestration script ⭐
├── whisper_with_vocab.py               # Stage 1: Vocabulary-enhanced Whisper
├── convert_aws_transcribe.py           # Stage 2: Format + SRT generation
├── claude_refine_transcript.py         # Stage 3: Claude AI refinement
├── apply_txt_corrections_to_srt.py     # Stage 4: Apply corrections to SRT
├── build_vocabulary.py                 # Build vocabulary files
├── keyword_lists/
│   ├── input.txt                       # Custom vocabulary (edit this!)
│   ├── whisper_vocabulary.json         # Auto-generated for Whisper
│   ├── replacement_map.json            # Auto-generated for Claude
│   ├── embedding_top_phrases_*.csv     # Common phrases (500)
│   └── embedding_top_words_*.csv       # Domain words (500)
├── downloads/                          # Output directory (default)
├── requirements.txt
├── README.md
├── PIPELINE.md                         # Detailed documentation ⭐
└── LICENSE
```

## Dependencies

Key libraries (see `requirements.txt` for full list):
- `openai-whisper==20250625` - Speech transcription
- `torch==2.9.0+cu128` - PyTorch with CUDA
- `boto3==1.35.99` - AWS SDK for Bedrock
- `numpy==1.26.4` - Numerical operations
- `tqdm==4.67.1` - Progress bars

## Performance & Cost

**20-minute audio file on RTX 5070 Ti**:
- **Time**: ~5.5 minutes total
  - Stage 1 (Whisper): ~4 min
  - Stage 2 (Format): ~5 sec
  - Stage 3 (Claude): ~45 sec
  - Stage 4 (Apply): ~2 sec
- **Cost**: ~$0.05 per file (Claude API only)
- **Quality**: Production-ready with proper paragraphs

**Scaling**:
- 100 hours of audio = ~$15-30 in Claude costs
- GPU required for reasonable Whisper speed
- Can process multiple files in parallel

## Custom Vocabulary

Add your own terms to `keyword_lists/input.txt`:

```
# Format: Display_Name	IPA-pronunciation	alternatives	Final Display
Ouspensky	oo-SPEN-skee	lispensky uspensky	Ouspensky
Maharaj-ji	muh-hah-RAHJ-jee	maharaji maraj ji	Maharaj-ji
Grey's-Anatomy	grayz uh-NAT-uh-mee		Grey's Anatomy
```

Then rebuild:
```powershell
python build_vocabulary.py
```

The pipeline will now recognize these terms in audio!

## Troubleshooting

- **CUDA out of memory**: Use smaller model (`--model small`)
- **AWS credentials error**: Run `aws configure` with Bedrock-enabled credentials
- **Claude timeout**: Large files (>2 hours) may need splitting
- **No paragraph breaks**: Lower threshold in `convert_aws_transcribe.py` (line 113)
- **Unicode errors**: Already fixed in current version (uses ASCII output)

See **[PIPELINE.md](PIPELINE.md)** for detailed troubleshooting and advanced usage.

## Documentation

- **[README.md](README.md)** - This file (quick start)
- **[PIPELINE.md](PIPELINE.md)** - Comprehensive documentation (architecture, scripts, troubleshooting)

## License

MIT License - see [LICENSE](LICENSE) file for details

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for state-of-the-art transcription
- [Anthropic Claude](https://www.anthropic.com/) for intelligent refinement
- Ram Dass Foundation for preserving the teachings

---

**Made with ❤️ for the Ram Dass community**
