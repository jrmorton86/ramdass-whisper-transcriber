# Ramdass Whisper Transcriber

Transcribe audio files with speaker diarization using OpenAI Whisper and pyannote.audio. Designed for transcribing Ram Dass lectures and multi-speaker recordings with automatic speaker identification.

## Features

- 🎙️ High-quality transcription using OpenAI Whisper
- 👥 Speaker diarization with pyannote.audio
- 🏷️ Automatic speaker name mapping
- 📊 GPU acceleration support
- 📝 Multiple output formats (JSON, formatted text, simple text)
- ⚙️ Configurable speaker detection parameters

## Prerequisites

- Python 3.12+
- FFmpeg (for audio processing)
- CUDA-capable GPU (optional, for faster processing)

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

4. **Set up HuggingFace Token** (for speaker diarization):
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with read access
   - Accept the model terms at: https://huggingface.co/pyannote/speaker-diarization-community-1
   - Create a `.env` file in the project root:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your token:
     ```
     HF_TOKEN=your_token_here
     ```
   - Alternatively, set as an environment variable:
     ```powershell
     $env:HF_TOKEN = "your_token_here"
     ```

## Usage

### Basic Transcription (without speaker diarization)

```powershell
python transcribe_with_diarization.py "downloads\Clip 1_1969.mp3"
```

This will work without a HuggingFace token but will label all segments as SPEAKER_00.

### With Speaker Diarization

```powershell
python transcribe_with_diarization.py "downloads\Clip 1_1969.mp3" --hf-token YOUR_TOKEN
```

Or set the environment variable:

```powershell
$env:HF_TOKEN = "your_token_here"
python transcribe_with_diarization.py "downloads\Clip 1_1969.mp3"
```

### With Speaker Identification

Automatically identify speakers by name instead of generic labels:

```powershell
# Map detected speakers to names in order
python transcribe_with_diarization.py "downloads\Clip 1_1969.mp3" --speaker-names "Ram Dass,Host,Caller"

# Use a speaker configuration file for more control
python transcribe_with_diarization.py "downloads\Clip 1_1969.mp3" --speaker-config speakers.json
```

The script will map `SPEAKER_00` → "Ram Dass", `SPEAKER_01` → "Host", etc.

### Options

- `-m, --model`: Whisper model size (tiny, base, small, medium, large)
  - Default: `base`
  - Larger models are more accurate but slower
  - Example: `python transcribe_with_diarization.py audio.mp3 -m medium`

- `-o, --output-dir`: Output directory
  - Default: same directory as audio file
  - Example: `python transcribe_with_diarization.py audio.mp3 -o transcripts`

- `--speaker-names`: Comma-separated list of speaker names
  - Maps detected speakers to real names in order
  - Example: `--speaker-names "Ram Dass,Host,Caller"`
  
- `--speaker-config`: Path to speaker configuration JSON file
  - For advanced speaker identification with aliases and characteristics
  - Example: `--speaker-config speakers.json`

- `--min-speakers`, `--max-speakers`: Control number of detected speakers
  - Helps diarization when you know the speaker count
  - Example: `--min-speakers 2 --max-speakers 3`

## Output Files

The script generates three files:

1. **`{filename}_with_speakers.json`**: Detailed JSON with timestamps and speaker labels
2. **`{filename}_with_speakers.txt`**: Formatted transcript with timestamps
3. **`{filename}_simple.txt`**: Clean transcript grouped by speaker

## Example Workflow

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Single file with speaker identification
python transcribe_with_diarization.py "audio.mp3" --speaker-names "Ram Dass,Host" -m medium

# Batch process multiple files
Get-ChildItem .\audio\*.mp3 | ForEach-Object { 
    python transcribe_with_diarization.py $_.FullName --speaker-names "Ram Dass,Host,Caller" 
}
```

## Project Structure

```
ramdass-whisper-transcriber/
├── transcribe_with_diarization.py  # Main transcription script
├── speaker_config.py               # Speaker identification system
├── simple_diarization.py          # Basic diarization (Windows fallback)
├── test_diarization.py            # Testing utilities
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
├── speakers_example.json          # Example speaker configuration
├── README.md
└── LICENSE
```

## Dependencies

Key libraries (see `requirements.txt` for full list):
- `openai-whisper==20250625` - Speech transcription
- `pyannote.audio==4.0.1` - Speaker diarization
- `torch==2.9.0` - PyTorch for model inference
- `python-dotenv==1.2.1` - Environment variable management
- `tqdm==4.67.1` - Progress bars

## Performance Tips

- **GPU Acceleration**: Automatically used if CUDA is available (10-20x faster)
- **Model Selection**: 
  - `tiny` - Fastest, lower accuracy (~1x realtime on CPU)
  - `base` - Good balance (default, ~3x realtime on CPU)
  - `medium` - Better accuracy (~10x realtime on CPU)
  - `large` - Best accuracy (~20x realtime on CPU, 6GB+ VRAM)
- **Speaker Count**: Provide `--min-speakers` and `--max-speakers` for better diarization accuracy

## Troubleshooting

- **CUDA out of memory**: Use a smaller Whisper model or process shorter audio segments
- **Diarization errors**: Check HuggingFace token and model terms acceptance
- **Audio format issues**: Ensure FFmpeg is installed and accessible in PATH

## License

MIT License - see [LICENSE](LICENSE) file for details

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for state-of-the-art transcription
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization
- Ram Dass Foundation for preserving the teachings
