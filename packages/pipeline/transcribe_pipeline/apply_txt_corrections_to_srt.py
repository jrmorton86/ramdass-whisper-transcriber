#!/usr/bin/env python3
"""
Apply TXT Corrections to SRT

Takes the corrections from claude_refine_transcript.py and applies them to the SRT file.
This avoids making a second expensive Claude API call.

Usage:
    python apply_txt_corrections_to_srt.py <changes_json> <srt_file>
"""

import argparse
import json
from pathlib import Path


def parse_srt(srt_content):
    """Parse SRT file into structured subtitles"""
    subtitles = []
    blocks = srt_content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                index = int(lines[0])
                timing = lines[1]
                text = '\n'.join(lines[2:])
                subtitles.append({
                    'index': index,
                    'timing': timing,
                    'text': text
                })
            except (ValueError, IndexError):
                continue
    
    return subtitles


def subtitles_to_srt(subtitles):
    """Convert subtitle list back to SRT format"""
    srt_lines = []
    for sub in subtitles:
        srt_lines.append(str(sub['index']))
        srt_lines.append(sub['timing'])
        srt_lines.append(sub['text'])
        srt_lines.append('')  # Blank line between subtitles
    
    return '\n'.join(srt_lines)


def apply_corrections_to_srt(subtitles, corrections):
    """Apply corrections from TXT refinement to SRT subtitles"""
    changes_made = []
    
    for correction in corrections:
        # Extract the text change (ignore paragraph breaks for SRT)
        original = correction.get('original', '')
        corrected = correction.get('corrected', '')
        
        if not original or not corrected:
            continue
        
        # Skip if this is just adding paragraph breaks
        if original.strip() == corrected.strip():
            continue
        
        # Try to find and apply in individual subtitles first
        found = False
        for sub in subtitles:
            if original in sub['text']:
                sub['text'] = sub['text'].replace(original, corrected, 1)
                found = True
                changes_made.append({
                    **correction,
                    'subtitle_index': sub['index']
                })
                # Use ASCII-safe output to avoid encoding errors
                orig_safe = original[:50].encode('ascii', 'replace').decode('ascii')
                corr_safe = corrected[:50].encode('ascii', 'replace').decode('ascii')
                print(f"  [OK] Applied to subtitle #{sub['index']}: {orig_safe}... -> {corr_safe}...")
                break
        
        if not found:
            # Try across multiple subtitles (concatenated)
            full_text = " ".join([sub['text'] for sub in subtitles])
            if original in full_text:
                # Find which subtitle(s) contain this text
                for sub in subtitles:
                    # Try to match parts of the correction
                    words = original.split()
                    if len(words) > 3:
                        # Try matching a portion
                        for i in range(len(words) - 2):
                            partial = ' '.join(words[i:i+3])
                            if partial in sub['text']:
                                # This subtitle might be affected
                                old_text = sub['text']
                                # Try to intelligently replace
                                for j in range(len(words)):
                                    attempt = ' '.join(words[j:])
                                    if attempt in sub['text']:
                                        corresponding_corrected = ' '.join(corrected.split()[j:])
                                        sub['text'] = sub['text'].replace(attempt, corresponding_corrected, 1)
                                        if sub['text'] != old_text:
                                            changes_made.append({
                                                **correction,
                                                'subtitle_index': sub['index']
                                            })
                                            print(f"  [OK] Partially applied to subtitle #{sub['index']}")
                                            found = True
                                            break
                                if found:
                                    break
                        if found:
                            break
        
        if not found:
            # Use ASCII-safe output to avoid encoding errors
            orig_safe = original[:50].encode('ascii', 'replace').decode('ascii')
            print(f"  [SKIP] Could not apply: {orig_safe}...")
    
    return changes_made


def apply_corrections_to_srt_file(changes_json_path, srt_path, output_path=None):
    """
    Apply TXT corrections to SRT file.

    Designed for worker pools - takes file paths directly.

    Args:
        changes_json_path: Path to changes JSON from claude_refine_transcript
        srt_path: Path to SRT file to refine
        output_path: Optional output path (default: strips _temp suffix, e.g., base_temp.srt -> base.srt)

    Returns:
        dict with 'srt_path', 'changes_path', 'changes_count'
    """
    changes_path = Path(changes_json_path)
    srt_file = Path(srt_path)

    # Load changes
    with open(changes_path, 'r', encoding='utf-8') as f:
        changes_data = json.load(f)

    corrections = changes_data.get('changes', [])
    print(f"Loaded {len(corrections)} corrections")

    # Load and parse SRT
    with open(srt_file, 'r', encoding='utf-8') as f:
        srt_content = f.read()

    subtitles = parse_srt(srt_content)
    print(f"Loaded {len(subtitles)} subtitles")

    # Apply corrections
    changes_made = apply_corrections_to_srt(subtitles, corrections)
    print(f"Applied {len(changes_made)} corrections")

    # Determine output path
    if output_path:
        out_path = Path(output_path)
    else:
        # Strip _temp suffix from input filename for output (e.g., base_temp.srt -> base.srt)
        base_stem = srt_file.stem.removesuffix('_temp')
        out_path = srt_file.parent / f"{base_stem}.srt"

    # Save refined SRT
    refined_srt = subtitles_to_srt(subtitles)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(refined_srt)
    print(f"[OK] Saved refined SRT: {out_path}")

    # Save changes log
    changes_log_path = out_path.parent / f"{out_path.stem}_changes.json"
    with open(changes_log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'changes': changes_made,
            'summary': {
                'total_subtitles': len(subtitles),
                'total_changes': len(changes_made)
            }
        }, f, indent=2, ensure_ascii=False)

    return {
        'srt_path': str(out_path),
        'changes_path': str(changes_log_path),
        'changes_count': len(changes_made)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Apply TXT corrections to SRT file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply corrections from TXT refinement to SRT
  python apply_txt_corrections_to_srt.py downloads/Clip_1_logs.json downloads/Clip_1_temp.srt
        """
    )
    parser.add_argument('changes_json', help='Path to changes JSON from claude_refine_transcript.py')
    parser.add_argument('srt_file', help='Path to SRT file to refine')
    parser.add_argument('-o', '--output', help='Output SRT file (default: strips _temp suffix, e.g., base_temp.srt -> base.srt)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    changes_path = Path(args.changes_json)
    srt_path = Path(args.srt_file)
    
    if not changes_path.exists():
        print(f"[ERROR] Changes file not found: {changes_path}")
        return 1
    
    if not srt_path.exists():
        print(f"[ERROR] SRT file not found: {srt_path}")
        return 1
    
    # Load changes
    print(f"\n{'='*70}")
    print(f"APPLYING TXT CORRECTIONS TO SRT")
    print(f"{'='*70}\n")
    print(f"Changes JSON: {changes_path}")
    print(f"SRT file:     {srt_path}")
    
    with open(changes_path, 'r', encoding='utf-8') as f:
        changes_data = json.load(f)
    
    corrections = changes_data.get('changes', [])
    print(f"\nLoaded {len(corrections)} corrections from TXT refinement")
    
    # Load SRT
    with open(srt_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    
    subtitles = parse_srt(srt_content)
    print(f"Loaded {len(subtitles)} subtitles from SRT")
    
    # Apply corrections
    print(f"\nApplying corrections to SRT...")
    changes_made = apply_corrections_to_srt(subtitles, corrections)
    
    print(f"\n{'='*70}")
    print(f"Applied {len(changes_made)} corrections to SRT")
    print(f"{'='*70}")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Strip _temp suffix from input filename for output (e.g., base_temp.srt -> base.srt)
        base_stem = srt_path.stem.removesuffix('_temp')
        output_path = srt_path.parent / f"{base_stem}.srt"
    
    # Save refined SRT
    refined_srt = subtitles_to_srt(subtitles)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(refined_srt)
    
    print(f"\n[OK] Saved refined SRT: {output_path}")
    
    # Save changes log
    changes_log_path = output_path.parent / f"{output_path.stem}_changes.json"
    with open(changes_log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'changes': changes_made,
            'summary': {
                'total_subtitles': len(subtitles),
                'total_changes': len(changes_made),
                'source_changes': len(corrections)
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved changes log: {changes_log_path}\n")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
