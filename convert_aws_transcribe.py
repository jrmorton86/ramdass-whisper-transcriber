#!/usr/bin/env python3
"""
Convert Whisper or AWS Transcribe JSON to SRT and formatted TXT files.

This script takes either:
- Whisper JSON output (from whisper_with_vocab.py)
- AWS Transcribe JSON output (with speaker diarization)

And generates:
1. SRT subtitle file with timecodes (and speaker labels for AWS Transcribe)
2. Formatted TXT file with intelligent paragraph breaks based on timing analysis

Usage:
    python convert_aws_transcribe.py <json_file> [--output-dir <dir>] [--format whisper|aws]

Example:
    python convert_aws_transcribe.py downloads/Clip_1_1969.json
    python convert_aws_transcribe.py downloads/transcript.json --format aws
    python convert_aws_transcribe.py downloads/Clip_1_1969.json --output-dir output/
"""

import argparse
import json
import os
import re
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional


def detect_format(transcript_json: Dict) -> str:
    """
    Detect if JSON is from Whisper or AWS Transcribe.
    
    Args:
        transcript_json: The JSON data
        
    Returns:
        'whisper' or 'aws'
    """
    # Whisper has 'segments' with 'id', 'seek', 'start', 'end', 'text'
    # AWS Transcribe has 'results' with 'items' and 'speaker_labels'
    
    if 'results' in transcript_json and 'items' in transcript_json.get('results', {}):
        return 'aws'
    elif 'segments' in transcript_json and 'text' in transcript_json:
        return 'whisper'
    else:
        raise ValueError("Unknown JSON format - not Whisper or AWS Transcribe")


def format_srt_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        SRT formatted timestamp
    """
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


# ============================================================================
# WHISPER FORMAT CONVERTERS
# ============================================================================

def whisper_to_srt(transcript_json: Dict) -> str:
    """
    Convert Whisper JSON to SRT format with timecodes.
    
    Args:
        transcript_json: Whisper output JSON with 'segments'
        
    Returns:
        SRT formatted string
    """
    print("Converting Whisper transcript to SRT format...")
    
    segments = transcript_json.get('segments', [])
    if not segments:
        print("WARNING:  No segments found in Whisper JSON")
        return ""
    
    srt_entries = []
    
    for i, segment in enumerate(segments, 1):
        start_time = segment.get('start', 0.0)
        end_time = segment.get('end', 0.0)
        text = segment.get('text', '').strip()
        
        if not text:
            continue
        
        # Format timestamps as SRT (HH:MM:SS,mmm)
        start_srt = format_srt_timestamp(start_time)
        end_srt = format_srt_timestamp(end_time)
        
        # Create SRT entry
        srt_entry = f"{i}\n{start_srt} --> {end_srt}\n{text}\n"
        srt_entries.append(srt_entry)
    
    print(f"  Generated {len(srt_entries)} subtitle entries")
    return '\n'.join(srt_entries)


def whisper_to_formatted_text(transcript_json: Dict, max_pause: float = 1.5) -> str:
    """
    Convert Whisper JSON to formatted text with intelligent paragraph breaks.
    Uses timing gaps between segments to determine paragraph boundaries.
    
    Args:
        transcript_json: Whisper output JSON with 'segments'
        max_pause: Minimum pause (in seconds) before starting new paragraph (default: 1.5)
        
    Returns:
        Formatted text with paragraph breaks
    """
    print("Converting Whisper transcript to formatted text with intelligent paragraphing...")
    
    segments = transcript_json.get('segments', [])
    if not segments:
        print("WARNING:  No segments found in Whisper JSON")
        return ""
    
    # Calculate timing gaps between segments
    for i in range(1, len(segments)):
        prev_end = segments[i-1].get('end', 0.0)
        curr_start = segments[i].get('start', 0.0)
        segments[i]['pause_before'] = max(0.0, curr_start - prev_end)
    segments[0]['pause_before'] = 0.0
    
    # Calculate statistics for pause threshold
    pauses = [s['pause_before'] for s in segments[1:] if s['pause_before'] > 0.1]
    
    if pauses:
        pauses_sorted = sorted(pauses)
        n = len(pauses_sorted)
        median = pauses_sorted[n // 2]
        q3 = pauses_sorted[3 * n // 4]
        
        # Use Q3 (75th percentile) for less sensitivity to pauses
        # This means only the longest 25% of pauses will trigger paragraph breaks
        pause_threshold = max(q3, max_pause)
        print(f"  Pause threshold: {pause_threshold:.2f}s (Q3: {q3:.2f}s, median: {median:.2f}s)")
    else:
        pause_threshold = max_pause
    
    # Build paragraphs based on pauses
    paragraphs = []
    current_paragraph = []
    
    for segment in segments:
        text = segment.get('text', '').strip()
        if not text:
            continue
        
        # Start new paragraph if pause is significant
        if segment['pause_before'] > pause_threshold and current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []
        
        current_paragraph.append(text)
    
    # Add final paragraph
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    print(f"  Generated {len(paragraphs)} paragraphs from {len(segments)} segments")
    
    # Format output with proper spacing
    return '\n\n'.join(paragraphs)


# ============================================================================
# AWS TRANSCRIBE FORMAT CONVERTERS
# ============================================================================

def transcribe_to_srt(transcript_json: Dict) -> str:
    """
    Convert AWS Transcribe JSON to SRT format with timecodes.
    
    Args:
        transcript_json: AWS Transcribe output JSON
        
    Returns:
        SRT formatted string
    """
    print("Converting transcript to SRT format...")
    
    items = transcript_json['results']['items']
    speaker_segments = transcript_json['results'].get('speaker_labels', {}).get('segments', [])
    
    srt_entries = []
    entry_num = 1
    
    for segment in speaker_segments:
        speaker = segment.get('speaker_label', 'SPEAKER_00')
        start_time = float(segment['start_time'])
        end_time = float(segment['end_time'])
        
        # Get words for this segment
        segment_items = segment.get('items', [])
        words = []
        
        for item in segment_items:
            # Find the word in items
            for word_item in items:
                if word_item.get('start_time') == item.get('start_time'):
                    words.append(word_item.get('alternatives', [{}])[0].get('content', ''))
                    break
        
        text = ' '.join(words)
        
        # Format timestamps as SRT (HH:MM:SS,mmm)
        start_srt = format_srt_timestamp(start_time)
        end_srt = format_srt_timestamp(end_time)
        
        # Create SRT entry
        srt_entry = f"{entry_num}\n{start_srt} --> {end_srt}\n{speaker}: {text}\n"
        srt_entries.append(srt_entry)
        entry_num += 1
    
    return '\n'.join(srt_entries)


def transcribe_to_formatted_text(transcript_json: Dict) -> Optional[str]:
    """
    Convert AWS Transcribe JSON to formatted text with intelligent paragraph breaks.
    Uses statistical analysis of sentence timing delays to determine natural paragraph boundaries.
    
    Args:
        transcript_json: AWS Transcribe output JSON
        
    Returns:
        Formatted text with speaker labels and paragraph breaks, or None if empty
    """
    print("Converting transcript to formatted text with intelligent paragraphing...")
    
    # Check if transcript has valid content
    if not transcript_json or not transcript_json.get('results'):
        print("WARNING:  Transcript is empty or has no results - likely no speech content in audio")
        return None
    
    items = transcript_json['results'].get('items', [])
    if not items:
        print("WARNING:  No items found in transcript - audio contains no speech")
        return None
    
    speaker_segments = transcript_json['results'].get('speaker_labels', {}).get('segments', [])
    
    # Step 1: Build sentences with metadata
    sentences = []
    current_sentence = []
    current_start = None
    current_end = None
    current_speaker = None
    
    for item in items:
        if item['type'] == 'pronunciation':
            word = item['alternatives'][0]['content']
            start_time = float(item.get('start_time', 0))
            end_time = float(item.get('end_time', 0))
            
            # Find speaker for this timestamp
            speaker = 'SPEAKER_00'
            for segment in speaker_segments:
                seg_start = float(segment['start_time'])
                seg_end = float(segment['end_time'])
                if seg_start <= start_time <= seg_end:
                    speaker = segment['speaker_label']
                    break
            
            if current_start is None:
                current_start = start_time
            current_end = end_time
            current_speaker = speaker
            current_sentence.append(word)
            
        elif item['type'] == 'punctuation':
            punct = item['alternatives'][0]['content']
            current_sentence.append(punct)
            
            # Sentence boundary markers
            if punct in ['.', '!', '?']:
                if current_sentence:
                    # Join words with spaces, then fix punctuation spacing
                    text = ' '.join(current_sentence)
                    text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
                    text = text.replace(' ;', ';').replace(' :', ':')
                    
                    sentences.append({
                        'text': text,
                        'start_time': current_start,
                        'end_time': current_end,
                        'speaker': current_speaker,
                        'word_count': len([w for w in current_sentence if w not in ['.', ',', '!', '?', ';', ':']])
                    })
                    current_sentence = []
                    current_start = None
                    current_end = None
    
    # Add any remaining sentence
    if current_sentence:
        # Join words with spaces, then fix punctuation spacing
        text = ' '.join(current_sentence)
        text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
        text = text.replace(' ;', ';').replace(' :', ':')
        
        sentences.append({
            'text': text,
            'start_time': current_start or 0,
            'end_time': current_end or 0,
            'speaker': current_speaker or 'SPEAKER_00',
            'word_count': len([w for w in current_sentence if w not in ['.', ',', '!', '?', ';', ':']])
        })
    
    if not sentences:
        return ""
    
    # Step 2: Calculate start delays between sentences
    for i in range(1, len(sentences)):
        delay = sentences[i]['start_time'] - sentences[i-1]['end_time']
        sentences[i]['start_delay'] = max(0, delay)
    sentences[0]['start_delay'] = 0
    
    # Step 3: Statistical analysis of delays (using median + IQR approach)
    delays = [s['start_delay'] for s in sentences[1:] if s['start_delay'] > 0]
    
    if delays:
        delays_sorted = sorted(delays)
        n = len(delays_sorted)
        median = delays_sorted[n // 2]
        q1 = delays_sorted[n // 4]
        q3 = delays_sorted[3 * n // 4]
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        
        # Use median as threshold (adjustable strategy)
        delay_threshold = max(median, 1.0)  # At least 1 second
        print(f"  Delay threshold: {delay_threshold:.2f}s (median: {median:.2f}s)")
    else:
        delay_threshold = 2.0  # Default fallback
    
    # Step 4: Mark paragraph breaks based on delays and speaker changes
    for i, sentence in enumerate(sentences):
        if i == 0:
            sentence['is_paragraph_start'] = True
        elif sentence['speaker'] != sentences[i-1]['speaker']:
            sentence['is_paragraph_start'] = True
        elif sentence['start_delay'] > delay_threshold:
            sentence['is_paragraph_start'] = True
        else:
            sentence['is_paragraph_start'] = False
    
    # Step 5: Build initial paragraphs
    paragraphs = []
    current_para_sentences = []
    current_para_speaker = None
    
    for sentence in sentences:
        if sentence['is_paragraph_start'] and current_para_sentences:
            # Save current paragraph
            paragraphs.append({
                'speaker': current_para_speaker,
                'sentences': current_para_sentences[:],
                'word_count': sum(s['word_count'] for s in current_para_sentences)
            })
            current_para_sentences = []
        
        current_para_speaker = sentence['speaker']
        current_para_sentences.append(sentence)
    
    # Add final paragraph
    if current_para_sentences:
        paragraphs.append({
            'speaker': current_para_speaker,
            'sentences': current_para_sentences[:],
            'word_count': sum(s['word_count'] for s in current_para_sentences)
        })
    
    # Step 6: Split overly long paragraphs
    word_counts = [p['word_count'] for p in paragraphs]
    if word_counts:
        wc_sorted = sorted(word_counts)
        n = len(wc_sorted)
        q3_words = wc_sorted[3 * n // 4] if n >= 4 else 150
        iqr_words = wc_sorted[3 * n // 4] - wc_sorted[n // 4] if n >= 4 else 50
        max_words = q3_words + 1.5 * iqr_words
        max_words = max(max_words, 200)  # At least 200 words
        
        print(f"  Max words per paragraph: {max_words:.0f}")
        
        split_paragraphs = []
        for para in paragraphs:
            if para['word_count'] > max_words and len(para['sentences']) > 1:
                # Find sentence with max delay within paragraph
                max_delay_idx = 0
                max_delay = 0
                for i, sent in enumerate(para['sentences'][1:], 1):
                    if sent['start_delay'] > max_delay:
                        max_delay = sent['start_delay']
                        max_delay_idx = i
                
                if max_delay_idx > 0:
                    # Split at that point
                    split_paragraphs.append({
                        'speaker': para['speaker'],
                        'sentences': para['sentences'][:max_delay_idx],
                        'word_count': sum(s['word_count'] for s in para['sentences'][:max_delay_idx])
                    })
                    split_paragraphs.append({
                        'speaker': para['speaker'],
                        'sentences': para['sentences'][max_delay_idx:],
                        'word_count': sum(s['word_count'] for s in para['sentences'][max_delay_idx:])
                    })
                else:
                    split_paragraphs.append(para)
            else:
                split_paragraphs.append(para)
        
        paragraphs = split_paragraphs
    
    # Step 7: Format output
    output_lines = []
    current_speaker_label = None
    
    for para in paragraphs:
        speaker = para['speaker']
        speaker_num = int(speaker.split('_')[-1]) + 1
        speaker_label = f"Speaker {speaker_num}"
        
        # Add speaker label if changed
        if speaker_label != current_speaker_label:
            if output_lines:  # Add blank line before new speaker
                output_lines.append("")
            output_lines.append(f"{speaker_label}:")
            current_speaker_label = speaker_label
        
        # Build paragraph text
        para_text = ' '.join(s['text'] for s in para['sentences'])
        output_lines.append(para_text)
        output_lines.append("")  # Blank line after paragraph
    
    print(f"  Generated {len(paragraphs)} paragraphs from {len(sentences)} sentences")
    
    return '\n'.join(output_lines)


def convert_transcript(json_file: str, output_dir: Optional[str] = None, force_format: Optional[str] = None):
    """
    Convert Whisper or AWS Transcribe JSON to SRT and formatted TXT files.
    Auto-detects format unless specified.
    
    Args:
        json_file: Path to JSON file (Whisper or AWS Transcribe)
        output_dir: Optional output directory (defaults to same directory as input)
        force_format: Force format detection ('whisper' or 'aws', default: auto-detect)
    """
    # Load JSON file
    json_path = Path(json_file)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    
    print(f"Loading transcript JSON: {json_file}")
    with open(json_path, 'r', encoding='utf-8') as f:
        transcript_json = json.load(f)
    
    # Detect format
    if force_format:
        format_type = force_format.lower()
        print(f"Using forced format: {format_type}")
    else:
        format_type = detect_format(transcript_json)
        print(f"Auto-detected format: {format_type}")
    
    # Determine output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = json_path.parent
    
    # Get base filename (without extension)
    base_filename = json_path.stem
    
    # Generate SRT file
    print("\n" + "="*60)
    if format_type == 'whisper':
        srt_content = whisper_to_srt(transcript_json)
    else:
        srt_content = transcribe_to_srt(transcript_json)
    
    srt_file = output_path / f"{base_filename}.srt"
    with open(srt_file, 'w', encoding='utf-8') as f:
        f.write(srt_content)
    print(f"[OK] SRT file saved: {srt_file}")
    
    # Generate formatted TXT file
    print("\n" + "="*60)
    if format_type == 'whisper':
        formatted_text = whisper_to_formatted_text(transcript_json)
    else:
        formatted_text = transcribe_to_formatted_text(transcript_json)
    
    if formatted_text:
        txt_file = output_path / f"{base_filename}_formatted.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(formatted_text)
        print(f"[OK] Formatted TXT file saved: {txt_file}")
    else:
        print("WARNING:  No text content generated (empty transcript)")
    
    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Convert Whisper or AWS Transcribe JSON to SRT and formatted TXT files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect format (Whisper or AWS Transcribe)
  python convert_aws_transcribe.py downloads/Clip_1_1969.json
  
  # Specify output directory
  python convert_aws_transcribe.py downloads/transcript.json --output-dir output/
  
  # Force format detection
  python convert_aws_transcribe.py downloads/transcript.json --format whisper
  python convert_aws_transcribe.py downloads/transcript.json --format aws
        """
    )
    parser.add_argument('json_file', help='Path to Whisper or AWS Transcribe JSON file')
    parser.add_argument('--output-dir', '-o', help='Output directory (default: same as input file)')
    parser.add_argument('--format', '-f', choices=['whisper', 'aws'], 
                        help='Force format (default: auto-detect)')
    
    args = parser.parse_args()
    
    try:
        convert_transcript(args.json_file, args.output_dir, args.format)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
