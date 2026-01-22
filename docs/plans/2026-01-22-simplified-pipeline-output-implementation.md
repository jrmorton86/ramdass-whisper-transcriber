# Simplified Pipeline Output Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce pipeline output from 6 files to 4 files with cleaner naming.

**Architecture:** Modify output paths in pipeline scripts, rename intermediate files with `_temp` suffix, add cleanup step at end.

**Tech Stack:** Python (existing pipeline scripts)

---

### Task 1: Update convert_aws_transcribe.py

**Files:**
- Modify: `packages/pipeline/transcribe_pipeline/convert_aws_transcribe.py`

**Changes:**
- Output SRT to `{base}_temp.srt` instead of `{base}.srt`
- This marks it as intermediate, to be replaced by refined version

---

### Task 2: Update claude_refine_transcript.py

**Files:**
- Modify: `packages/pipeline/transcribe_pipeline/claude_refine_transcript.py`

**Changes:**
- Output refined text to `{base}.txt` instead of `{base}_formatted_refined.txt`
- Output changes to `{base}_logs.json` instead of `{base}_formatted_refined_changes.json`

---

### Task 3: Update apply_txt_corrections_to_srt.py

**Files:**
- Modify: `packages/pipeline/transcribe_pipeline/apply_txt_corrections_to_srt.py`

**Changes:**
- Read from `{base}_temp.srt` (input)
- Read changes from `{base}_logs.json` (input)
- Output to `{base}.srt` instead of `{base}_refined.srt`

---

### Task 4: Update transcribe_pipeline.py

**Files:**
- Modify: `packages/pipeline/transcribe_pipeline/transcribe_pipeline.py`

**Changes:**
- Update file path variables to match new naming
- Add cleanup step after successful completion to delete:
  - `{base}_formatted.txt`
  - `{base}_temp.srt`
- Update success message to show only 4 files

---

### Task 5: Update pipeline_runner.py (API)

**Files:**
- Modify: `packages/api/app/services/pipeline_runner.py`

**Changes:**
- Update file path lookups in `process_file()`:
  - `refined_txt` → `{base}.txt`
  - `refined_srt` → `{base}.srt`
  - Remove references to `formatted_txt` and `original_srt`

---

### Task 6: End-to-end test

**Verification:**
- Run a test transcription through the API
- Verify only 4 files are created with correct names
- Verify intermediate files are cleaned up
