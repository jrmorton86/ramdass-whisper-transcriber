# Enhanced Whisper Transcription with Custom Vocabulary

This setup uses your AWS Transcribe vocabulary and keyword analysis to improve Whisper transcription accuracy.

## Quick Start

```bash
# Basic transcription (with vocabulary enhancement)
python whisper_with_vocab.py "path/to/audio.mp3"

# Use larger model for better accuracy
python whisper_with_vocab.py "path/to/audio.mp3" --model medium

# Remove filler words (uh, um, ugh)
python whisper_with_vocab.py "path/to/audio.mp3" --remove-fillers

# Custom output location
python whisper_with_vocab.py "path/to/audio.mp3" -o "output/transcript"
```

## How It Works

### 1. **Initial Prompt (Vocabulary Priming)**
Whisper's `initial_prompt` parameter biases transcription toward specific terms:
- Top 20 custom vocabulary terms (Ram Dass, Maharaj-ji, etc.)
- Domain-specific words from your keyword analysis
- Creates context: "Ram Dass, Maharaj-ji... teachings about dharma, meditation..."

### 2. **Post-Processing Corrections**
Uses `replacement_map.json` to fix common errors:
- `ramdas` → `Ram Dass`
- `maharaji` → `Maharaj-ji`
- `neem karoli baba` → `Neem Karoli Baba`
- 50+ other corrections from AWS vocabulary

### 3. **Filler Word Removal** (Optional)
Removes: uh, um, ugh based on `filter.txt`

## Files Used

- `keyword_lists/whisper_vocabulary.json` - Full vocabulary data
- `keyword_lists/replacement_map.json` - Post-processing corrections
- Output: `downloads/filename.txt` and `downloads/filename.json`

## Examples

```bash
# Transcribe test file
python whisper_with_vocab.py "downloads/test.wav"

# High-quality transcription with cleanup
python whisper_with_vocab.py "audio/lecture.mp3" --model large --remove-fillers

# Disable corrections (raw Whisper output)
python whisper_with_vocab.py "audio/lecture.mp3" --no-corrections
```

## Model Sizes

- `tiny` - Fastest, lowest accuracy (~1GB)
- `base` - Good balance (~1GB) ← **Default**
- `small` - Better accuracy (~2GB)
- `medium` - High accuracy (~5GB)
- `large` - Best accuracy (~10GB)

## Output Files

- `.txt` - Plain text transcription
- `.json` - Full details including:
  - Segments with timestamps
  - Original vs corrected text
  - Vocabulary enhancement metadata

## Tips

1. **First run**: Downloads Whisper model (~1-10GB depending on size)
2. **GPU**: Automatically uses CUDA if available (much faster!)
3. **Long files**: Use `medium` or `large` model for better accuracy
4. **Custom vocab**: Update `keyword_lists/input.txt` and run `build_vocabulary.py` to refresh
