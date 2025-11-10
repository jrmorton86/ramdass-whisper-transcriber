# Ram Dass Transcription Pipeline

A complete, production-ready transcription system for Ram Dass audio content. Features vocabulary-enhanced Whisper transcription, intelligent paragraphing, and AI-powered refinement with cost optimization.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Pipeline Architecture](#pipeline-architecture)
- [Core Scripts](#core-scripts)
- [Configuration](#configuration)
- [Cost & Performance](#cost--performance)
- [Troubleshooting](#troubleshooting)

---

## Overview

This pipeline transforms raw audio files into high-quality transcripts with:
- ✅ Vocabulary-enhanced transcription (custom spiritual/philosophical terms)
- ✅ Intelligent paragraph breaks for readability
- ✅ AI-powered error correction and refinement
- ✅ Synchronized SRT subtitle generation
- ✅ Cost-optimized (single Claude API call for both TXT and SRT)

### What Makes This Pipeline Special

1. **Vocabulary Enhancement**: 32,768 token prompts with 60 custom terms (Ouspensky, Maharaj-ji, Gurdjieff, etc.)
2. **Anti-Hallucination**: Whisper configured to prevent repetition and false content
3. **Intelligent Paragraphing**: Uses Q3 (75th percentile) of pause durations for natural breaks
4. **Cost Optimization**: Reuses TXT corrections for SRT (saves ~$0.05 per file)
5. **Claude Sonnet 4.5**: Latest model with 200K context window for large files

---

## Quick Start

### Prerequisites

```bash
# Python 3.12+ with virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials for Claude Bedrock
aws configure
# Region: us-east-1
# Access credentials with Bedrock permissions
```

### Basic Usage

```bash
# Full pipeline (all stages)
python transcribe_pipeline.py "audio.mp3" --model medium

# Skip Whisper (use existing JSON)
python transcribe_pipeline.py "audio.mp3" --skip-whisper

# Skip Claude refinement
python transcribe_pipeline.py "audio.mp3" --skip-claude

# Verbose output
python transcribe_pipeline.py "audio.mp3" --model large --verbose
```

### Output Files

After running the pipeline, you'll get:

```
downloads/
├── audio.json                              # Raw Whisper output (word-level timing)
├── audio.txt                               # Simple transcript
├── audio.srt                               # Original subtitles
├── audio_formatted.txt                     # With paragraph breaks
├── audio_formatted_refined.txt             # Final refined transcript ⭐
├── audio_formatted_refined_changes.json    # List of corrections made
├── audio_refined.srt                       # Final refined subtitles ⭐
└── audio_refined_changes.json              # SRT corrections log
```

---

## Pipeline Architecture

### 4-Stage Process

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Whisper Transcription                                  │
│ Input:  audio.mp3                                               │
│ Output: audio.json (word-level timing + text)                   │
│ Script: whisper_with_vocab.py                                   │
│ Time:   ~2-5 minutes (medium model, RTX GPU)                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Format & Generate SRT                                  │
│ Input:  audio.json                                              │
│ Output: audio_formatted.txt, audio.srt                          │
│ Script: convert_aws_transcribe.py                               │
│ Time:   ~5 seconds                                              │
│ Logic:  Q3-based paragraph detection (1.5s threshold)           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Claude TXT Refinement                                  │
│ Input:  audio_formatted.txt                                     │
│ Output: audio_formatted_refined.txt + changes.json              │
│ Script: claude_refine_transcript.py                             │
│ Time:   ~30-60 seconds                                          │
│ API:    Claude Sonnet 4.5 (AWS Bedrock)                         │
│ Cost:   ~$0.05-0.15 per file                                    │
│ Tasks:  - Fix vocabulary errors                                 │
│         - Fix grammar/transcription errors                      │
│         - Add proper paragraph breaks                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: Apply Corrections to SRT                               │
│ Input:  audio.srt + audio_formatted_refined_changes.json        │
│ Output: audio_refined.srt + audio_refined_changes.json          │
│ Script: apply_txt_corrections_to_srt.py                         │
│ Time:   ~2 seconds                                              │
│ API:    None (reuses Stage 3 corrections)                       │
│ Cost:   $0 (optimization!)                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Scripts

### 1. `transcribe_pipeline.py`

**Purpose**: Main orchestration script that runs all 4 stages

**Usage**:
```bash
python transcribe_pipeline.py <audio_file> [options]

Options:
  -m, --model {tiny,base,small,medium,large}
                        Whisper model size (default: medium)
  -o, --output-dir PATH Output directory (default: downloads/)
  --skip-whisper        Skip transcription (use existing JSON)
  --skip-claude         Skip Claude refinement
  -v, --verbose         Show detailed progress
```

**Examples**:
```bash
# Production quality
python transcribe_pipeline.py "Clip_1.mp3" --model large

# Re-run formatting and refinement only
python transcribe_pipeline.py "Clip_1.mp3" --skip-whisper

# Quick test (no AI refinement)
python transcribe_pipeline.py "test.mp3" --model tiny --skip-claude
```

---

### 2. `whisper_with_vocab.py`

**Purpose**: Stage 1 - Vocabulary-enhanced Whisper transcription

**Key Features**:
- 32,768 token vocabulary prompts (vs default 224)
- Anti-hallucination settings (`condition_on_previous_text=False`)
- CUDA/GPU acceleration
- Progress bar with timing estimates

**Configuration** (lines 202-212):
```python
condition_on_previous_text=False  # Prevents repetition
compression_ratio_threshold=2.4   # Detects gibberish
logprob_threshold=-1.0           # Confidence filtering
no_speech_threshold=0.6          # Silence detection
```

**Vocabulary Loading**:
- Reads `keyword_lists/whisper_vocabulary.json`
- Includes 60 custom terms (Ouspensky, Maharaj-ji, Gurdjieff, etc.)
- Falls back to English-only if vocabulary missing

**Usage**:
```bash
python whisper_with_vocab.py <audio_file> [options]

Options:
  -m, --model MODEL     Whisper model (default: medium)
  -o, --output PATH     Output JSON file
  --verbose             Show detailed progress
```

**Output Format** (JSON):
```json
{
  "text": "Full transcript...",
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "We have had a rash recently...",
      "words": [
        {"word": "We", "start": 0.0, "end": 0.12},
        {"word": "have", "start": 0.12, "end": 0.24}
      ]
    }
  ]
}
```

---

### 3. `convert_aws_transcribe.py`

**Purpose**: Stage 2 - Convert Whisper JSON to formatted text with intelligent paragraphing and SRT subtitles

**Key Features**:
- Q3-based paragraph detection (75th percentile of pause durations)
- Configurable pause threshold (default: 1.5 seconds)
- Generates synchronized SRT subtitles
- Preserves word-level timing

**Paragraph Logic** (lines 113-180):
```python
# Calculate pause durations between all words
pauses = [word2.start - word1.end for consecutive words]

# Use Q3 (75th percentile) as baseline
q3_pause = np.percentile(pauses, 75)

# Threshold: max(user_setting, q3_pause)
threshold = max(1.5, q3_pause)

# Insert \n\n when pause exceeds threshold
```

**Usage**:
```bash
python convert_aws_transcribe.py <whisper_json> [options]

Options:
  --output PATH         Output TXT file
  --srt PATH            Output SRT file
  --max-pause SECONDS   Paragraph threshold (default: 1.5)
```

**Example**:
```bash
# Default settings (1.5s threshold)
python convert_aws_transcribe.py audio.json

# More paragraphs (lower threshold)
python convert_aws_transcribe.py audio.json --max-pause 1.0

# Fewer paragraphs (higher threshold)
python convert_aws_transcribe.py audio.json --max-pause 3.0
```

**SRT Format**:
```
1
00:00:00,000 --> 00:00:05,200
We have had a rash recently, a rash of beings

2
00:00:05,200 --> 00:00:10,400
who have come to talk to me and who have said,
```

---

### 4. `claude_refine_transcript.py`

**Purpose**: Stage 3 - AI-powered transcript refinement with paragraphing

**Key Features**:
- Uses Claude Sonnet 4.5 (200K context window)
- Vocabulary-aware corrections
- Adds proper paragraph breaks
- Returns full refined transcript + change log
- Conservative editing (95%+ confidence threshold)

**Claude Configuration**:
```python
Model:       us.anthropic.claude-sonnet-4-5-20250929-v1:0
Max Tokens:  20,000 (full transcript output)
Temperature: 0.3 (conservative)
Timeout:     180 seconds
```

**Prompt Strategy**:
1. Load ALL 60 custom vocabulary terms
2. Load ALL 500 common phrases
3. Load ALL 500 domain-specific words
4. Load known error patterns from replacement map
5. Instruct Claude to:
   - Fix clear vocabulary errors
   - Fix grammar/transcription errors
   - Add paragraph breaks for readability
   - Preserve speaker's voice and style
   - Be EXTREMELY conservative (95%+ confidence)

**Correction Types**:
- ✅ Vocabulary: "lispensky" → "Ouspensky"
- ✅ Grammar: "psychologies attempts" → "psychology's attempts"
- ✅ Duplicates: "on on the basis" → "on the basis"
- ✅ Fragments: "is that most" → "The thing is that most"
- ✅ Transcription: "I shout, my son" → "I want you to help my son"

**Paragraphing Rules**:
- New topic/example → new paragraph
- Story transitions → new paragraph
- Abstract to concrete shift → new paragraph
- Typical length: 3-8 sentences

**Usage**:
```bash
python claude_refine_transcript.py <input_txt> [options]

Options:
  -o, --output PATH     Output file (default: input_refined.txt)
  -v, --verbose         Show token usage and cost
  --region REGION       AWS region (default: us-east-1)
```

**Output** (JSON change log):
```json
{
  "refined_transcript": "Full refined text with \\n\\n breaks...",
  "changes_made": [
    {
      "type": "correction",
      "original": "lispensky",
      "corrected": "Ouspensky",
      "reason": "Known vocabulary error"
    }
  ],
  "paragraph_count": 15
}
```

**Cost Estimate**:
- Input: ~3,000 tokens (with vocabulary context)
- Output: ~2,000 tokens (full transcript)
- Cost: ~$0.05-0.15 per file

---

### 5. `apply_txt_corrections_to_srt.py`

**Purpose**: Stage 4 - Apply TXT corrections to SRT (no additional API call)

**Key Features**:
- Reuses corrections from Stage 3
- Matches corrections to subtitle text
- Handles multi-subtitle corrections
- Ignores paragraph breaks (not needed in SRT)
- Cost: $0 (optimization!)

**Matching Logic**:
1. Try exact match in individual subtitles
2. If not found, try partial matches
3. Handle corrections spanning multiple subtitles
4. Skip paragraph-only changes

**Usage**:
```bash
python apply_txt_corrections_to_srt.py <changes_json> <srt_file> [options]

Options:
  -o, --output PATH     Output SRT file (default: input_refined.srt)
  -v, --verbose         Verbose output
```

**Example**:
```bash
python apply_txt_corrections_to_srt.py \
  downloads/audio_formatted_refined_changes.json \
  downloads/audio.srt
```

**Output** (change log):
```json
{
  "changes": [
    {
      "type": "correction",
      "original": "Lispensky",
      "corrected": "Ouspensky",
      "subtitle_index": 8
    }
  ],
  "summary": {
    "total_subtitles": 98,
    "total_changes": 10,
    "source_changes": 12
  }
}
```

---

### 6. `build_vocabulary.py`

**Purpose**: Build vocabulary files from custom input

**Input Format** (`keyword_lists/input.txt`):
```
# Tab-separated: display_as, sounds_like (IPA), alternative sounds, final display
Ouspensky	oo-SPEN-skee	lispensky uspensky	Ouspensky
Maharaj-ji	muh-hah-RAHJ-jee	maharaji maraj ji	Maharaj-ji
Gurdjieff	GURD-jeef	gurdjief gurjief	Gurdjieff
Grey's-Anatomy	grayz uh-NAT-uh-mee		Grey's Anatomy
```

**Output Files**:
- `whisper_vocabulary.json`: For Whisper prompts (32K tokens)
- `replacement_map.json`: For Claude corrections (wrong→right)
- `embedding_top_phrases_*.csv`: Common phrase list
- `embedding_top_words_*.csv`: Domain vocabulary list

**Usage**:
```bash
python build_vocabulary.py

# Rebuilds all vocabulary files from input.txt
```

**When to Rebuild**:
- After adding new custom terms to `input.txt`
- After updating phrase/word lists
- Before running pipeline on new content

---

## Configuration

### Vocabulary Files

**Location**: `keyword_lists/`

**Key Files**:
- `input.txt` - Custom vocabulary (edit this)
- `whisper_vocabulary.json` - Auto-generated, used by Whisper
- `replacement_map.json` - Auto-generated, used by Claude
- `embedding_top_phrases_*.csv` - Common phrases (500 entries)
- `embedding_top_words_*.csv` - Domain words (500 entries)

**Adding New Terms**:
1. Edit `keyword_lists/input.txt`
2. Add line: `Display-Name	IPA-pronunciation	alternative-spellings	Final Display`
3. Run: `python build_vocabulary.py`
4. Run pipeline: terms now recognized

**Example**:
```
# Add new term
Ramana-Maharshi	ruh-MAH-nuh muh-HAR-shee	ramana maharshi	Ramana Maharshi

# Rebuild
python build_vocabulary.py

# Use in pipeline
python transcribe_pipeline.py audio.mp3 --model medium
```

### AWS Bedrock Configuration

**Requirements**:
- AWS account with Bedrock access
- Claude Sonnet 4.5 model enabled in `us-east-1`
- IAM credentials with `bedrock:InvokeModel` permission

**Setup**:
```bash
# Configure AWS CLI
aws configure
AWS Access Key ID: YOUR_ACCESS_KEY
AWS Secret Access Key: YOUR_SECRET_KEY
Default region: us-east-1

# Test access
aws bedrock-runtime invoke-model \
  --model-id us.anthropic.claude-sonnet-4-5-20250929-v1:0 \
  --body '{"messages":[{"role":"user","content":[{"text":"Hello"}]}],"inferenceConfig":{"maxTokens":100}}' \
  --cli-binary-format raw-in-base64-out \
  output.json
```

**Timeout Settings** (in `claude_refine_transcript.py`):
```python
config = Config(
    read_timeout=180,      # 3 minutes for large files
    connect_timeout=10,
    retries={'max_attempts': 3}
)
```

### Whisper Model Selection

| Model  | Size   | VRAM   | Speed      | Quality    | Use Case                |
|--------|--------|--------|------------|------------|-------------------------|
| tiny   | 39 MB  | ~1 GB  | 32x faster | Lowest     | Testing only            |
| base   | 74 MB  | ~1 GB  | 16x faster | Low        | Quick drafts            |
| small  | 244 MB | ~2 GB  | 6x faster  | Good       | Balanced                |
| medium | 769 MB | ~5 GB  | 2x faster  | Very good  | **Recommended**         |
| large  | 1.5 GB | ~10 GB | 1x (base)  | Best       | Production (slow)       |

**Recommendation**: Use `medium` for production (best quality/speed tradeoff)

---

## Cost & Performance

### Performance Benchmarks

**Test File**: 20-minute Ram Dass lecture (RTX 5070 Ti Laptop GPU)

| Stage | Time      | Details                          |
|-------|-----------|----------------------------------|
| 1     | ~4 min    | Whisper medium model (CUDA)      |
| 2     | ~5 sec    | Format + SRT generation          |
| 3     | ~45 sec   | Claude API call                  |
| 4     | ~2 sec    | Apply corrections to SRT         |
| Total | ~5.5 min  | End-to-end pipeline              |

### Cost Breakdown

**Per-File Costs**:

| Stage | Service         | Cost/File    | Details                           |
|-------|-----------------|--------------|-----------------------------------|
| 1     | Local GPU       | $0           | Whisper runs locally              |
| 2     | Local CPU       | $0           | Pure Python processing            |
| 3     | Claude API      | ~$0.05-0.15  | Depends on transcript length      |
| 4     | Local CPU       | $0           | Reuses Stage 3 corrections        |
| Total |                 | ~$0.05-0.15  | Per audio file                    |

**Claude Pricing** (as of Nov 2025):
- Input: $0.003 per 1K tokens
- Output: $0.015 per 1K tokens

**Example Calculation** (20-min file):
```
Input tokens:  3,500 (transcript + vocabulary)
Output tokens: 2,000 (refined transcript)

Cost = (3,500 × $0.003 / 1,000) + (2,000 × $0.015 / 1,000)
     = $0.0105 + $0.030
     = $0.0405 (~$0.04 per file)
```

**Cost Optimization**:
- ✅ Single Claude call (not separate TXT + SRT calls)
- ✅ Reuse corrections for SRT (saves ~$0.04)
- ✅ Conservative editing (smaller output tokens)
- ✅ Total savings: ~50% vs original approach

### Scaling Estimates

**Processing 100 hours of audio**:
- Files: ~300 × 20-min clips
- Time: ~27.5 hours (parallel possible)
- Cost: ~$15-30 (Claude only)
- Storage: ~50 GB (all output files)

---

## Troubleshooting

### Common Issues

#### 1. Whisper Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
```bash
# Use smaller model
python transcribe_pipeline.py audio.mp3 --model small

# Or increase GPU memory by closing other apps
# Or use CPU (slower): set device="cpu" in whisper_with_vocab.py
```

#### 2. Claude API Timeout

**Error**: `Read timeout after 180 seconds`

**Solutions**:
- File too large (>2 hours): split into chunks
- Increase timeout in `claude_refine_transcript.py`:
  ```python
  config = Config(read_timeout=300)  # 5 minutes
  ```

#### 3. AWS Credentials Not Found

**Error**: `Unable to locate credentials`

**Solutions**:
```bash
# Configure AWS CLI
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

#### 4. No Paragraph Breaks

**Problem**: Formatted text has no `\n\n` breaks

**Solutions**:
```bash
# Lower the pause threshold
python convert_aws_transcribe.py audio.json --max-pause 1.0

# Or edit convert_aws_transcribe.py line 113:
max_pause: float = 1.0  # More breaks
max_pause: float = 3.0  # Fewer breaks
```

#### 5. Claude Refuses to Add Paragraphs

**Problem**: Refined text has no paragraph breaks

**Solution**: Already fixed in current version. Claude now explicitly instructed to add `\n\n` breaks.

#### 6. SRT Corrections Not Applied

**Problem**: `refined.srt` missing some corrections

**Cause**: Correction text doesn't exactly match subtitle text (due to subtitle splitting)

**Solution**: Review `*_refined_changes.json` to see which corrections were skipped. Most corrections should apply successfully.

#### 7. Unicode Encoding Errors (Windows)

**Error**: `UnicodeEncodeError: 'charmap' codec can't encode character`

**Solution**: Already fixed. All scripts now use ASCII for console output (`->` instead of `→`).

### Debugging Tips

**Verbose Mode**:
```bash
# See detailed progress and timing
python transcribe_pipeline.py audio.mp3 --verbose
```

**Check Intermediate Files**:
```bash
# Verify each stage output
ls -lh downloads/audio*

# Check JSON format
python -m json.tool downloads/audio.json | head -50

# Check changes made
python -m json.tool downloads/audio_formatted_refined_changes.json
```

**Test Individual Stages**:
```bash
# Test Whisper only
python whisper_with_vocab.py audio.mp3 --model medium

# Test formatting only
python convert_aws_transcribe.py audio.json

# Test Claude only
python claude_refine_transcript.py audio_formatted.txt --verbose

# Test SRT application only
python apply_txt_corrections_to_srt.py \
  audio_formatted_refined_changes.json \
  audio.srt
```

### Performance Optimization

**GPU Acceleration**:
```python
# Verify CUDA is available
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Your GPU name
```

**Parallel Processing**:
```bash
# Process multiple files (separate terminal for each)
python transcribe_pipeline.py file1.mp3 --model medium &
python transcribe_pipeline.py file2.mp3 --model medium &
python transcribe_pipeline.py file3.mp3 --model medium &

# Or use GNU parallel
ls *.mp3 | parallel python transcribe_pipeline.py {} --model medium
```

---

## Advanced Usage

### Batch Processing Script

Create `batch_process.sh`:
```bash
#!/bin/bash
for file in audio_files/*.mp3; do
    echo "Processing: $file"
    python transcribe_pipeline.py "$file" --model medium --verbose
    if [ $? -eq 0 ]; then
        echo "✓ Success: $file"
    else
        echo "✗ Failed: $file"
    fi
done
```

### Custom Vocabulary Workflow

1. **Extract terms from existing transcripts**:
   ```bash
   # Find frequent unknown words
   grep -oh '\w\+' downloads/*.txt | \
     sort | uniq -c | sort -rn | \
     head -100 > potential_vocab.txt
   ```

2. **Add to vocabulary**:
   ```bash
   # Edit keyword_lists/input.txt
   vim keyword_lists/input.txt
   
   # Rebuild
   python build_vocabulary.py
   ```

3. **Re-run pipeline**:
   ```bash
   # Use existing Whisper output, just re-format and re-refine
   python transcribe_pipeline.py audio.mp3 --skip-whisper
   ```

### Quality Assurance

**Manual Review Checklist**:
- [ ] Paragraph breaks are natural (not too many/few)
- [ ] Proper nouns spelled correctly
- [ ] No repeated phrases/words
- [ ] SRT timing looks correct
- [ ] Speaker's voice preserved

**Automated Checks**:
```python
# Check for obvious errors
import json

# Load changes
with open('audio_formatted_refined_changes.json') as f:
    changes = json.load(f)

# Count correction types
corrections = changes['changes']
print(f"Total corrections: {len(corrections)}")
print(f"Paragraph count: {changes['summary']['paragraph_count']}")

# Check for suspicious patterns
for c in corrections:
    if 'duplicate' in c['reason'].lower():
        print(f"Duplicate found: {c['original']}")
```

---

## Maintenance

### Regular Updates

**Weekly**:
- Review new audio files for unknown terms
- Add to `keyword_lists/input.txt`
- Rebuild vocabulary

**Monthly**:
- Check for new Whisper models
- Review Claude pricing changes
- Update documentation

### Backup Strategy

**Critical Files**:
```bash
# Backup vocabulary
tar -czf vocab_backup_$(date +%Y%m%d).tar.gz keyword_lists/

# Backup output
tar -czf transcripts_backup_$(date +%Y%m%d).tar.gz downloads/
```

---

## Support & Contributing

### Getting Help

1. Check this documentation first
2. Review error messages carefully
3. Test individual stages to isolate issues
4. Check AWS CloudWatch logs for Bedrock errors

### Known Limitations

- Max audio length: ~3 hours (due to Claude context window)
- Requires CUDA GPU for fast Whisper processing
- Requires AWS Bedrock access (paid service)
- English transcription only (can be extended)

---

## License

See `LICENSE` file for details.

## Changelog

### Version 2.0 (November 2025)
- ✅ Added intelligent paragraph generation by Claude
- ✅ Optimized SRT refinement (reuses TXT corrections)
- ✅ Removed unused scripts (8 files deleted)
- ✅ Fixed Windows Unicode encoding issues
- ✅ Cost reduction: 50% savings ($0.10 → $0.05 per file)
- ✅ Updated to Claude Sonnet 4.5

### Version 1.0 (November 2025)
- Initial production release
- 4-stage pipeline: Whisper → Format → Claude TXT → Apply to SRT
- Vocabulary enhancement with 60 custom terms
- Anti-hallucination settings for Whisper
