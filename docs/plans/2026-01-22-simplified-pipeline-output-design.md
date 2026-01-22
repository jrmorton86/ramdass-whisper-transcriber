# Simplified Pipeline Output Design

## Goal

Reduce pipeline output from 6 files to 4 files with cleaner naming convention.

## Current State (6 files)

1. `{uuid}.json` - Raw transcript
2. `{uuid}.srt` - Unrefined SRT (intermediate)
3. `{uuid}_formatted.txt` - Formatted text (intermediate)
4. `{uuid}_formatted_refined.txt` - Refined text
5. `{uuid}_refined.srt` - Refined subtitles
6. `{uuid}_formatted_refined_changes.json` - Change log

## New State (4 files)

1. `{uuid}.json` - Raw transcript
2. `{uuid}.txt` - Refined text
3. `{uuid}.srt` - Refined subtitles
4. `{uuid}_logs.json` - Change log

## File Flow

```
Audio File
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Stage 1: Whisper                                │
│   Output: {uuid}.json (KEEP)                    │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Stage 2: Format                                 │
│   Output: {uuid}_formatted.txt (TEMP)           │
│           {uuid}_temp.srt (TEMP)                │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Stage 3: Claude Refine                          │
│   Input:  {uuid}_formatted.txt                  │
│   Output: {uuid}.txt (KEEP)                     │
│           {uuid}_logs.json (KEEP)               │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Stage 4: Apply to SRT                           │
│   Input:  {uuid}_temp.srt, {uuid}_logs.json     │
│   Output: {uuid}.srt (KEEP)                     │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│ Cleanup                                         │
│   Delete: {uuid}_formatted.txt                  │
│           {uuid}_temp.srt                       │
└─────────────────────────────────────────────────┘
```

## Files to Modify

| File | Changes |
|------|---------|
| `transcribe_pipeline.py` | Update output paths, add cleanup, update success message |
| `claude_refine_transcript.py` | Output to `{base}.txt` and `{base}_logs.json` |
| `apply_txt_corrections_to_srt.py` | Output to `{base}.srt` |
| `convert_aws_transcribe.py` | Output SRT to `{base}_temp.srt` |
| `pipeline_runner.py` (API) | Update file path lookups for results |

## Naming Convention

- No `_formatted_refined` or `_refined` suffixes
- Only suffix is `_logs` for the change log
- Clean, predictable: `{uuid}.{ext}` for main outputs
