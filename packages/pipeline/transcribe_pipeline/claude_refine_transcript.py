#!/usr/bin/env python3
"""
Claude-powered transcript refinement using AWS Bedrock.

Uses Claude Opus 4.5 with the same vocabulary context given to Whisper
to fix transcription errors while being extremely conservative about changes.

Only fixes:
- Clear misspellings of terms explicitly in the vocabulary
- Grammar issues that are obvious errors
- Formatting improvements

Does NOT:
- Change words that are plausible/correct even if uncommon
- Add content that wasn't transcribed
- Hallucinate or embellish
"""

import json
import re
import os
import boto3
from pathlib import Path
import argparse
import time


class ModelStreamError(Exception):
    """Raised when Claude streaming encounters a model error (retryable)."""
    pass


class ClaudeTranscriptRefiner:
    def __init__(self,
                 vocab_file="keyword_lists/whisper_vocabulary.json",
                 replacement_map_file="keyword_lists/replacement_map.json",
                 region_name="us-east-1"):
        # Make paths relative to this script's directory
        script_dir = Path(__file__).parent
        self.vocab_file = script_dir / vocab_file if not Path(vocab_file).is_absolute() else Path(vocab_file)
        self.replacement_map_file = script_dir / replacement_map_file if not Path(replacement_map_file).is_absolute() else Path(replacement_map_file)
        self.vocab_data = None
        self.replacement_map = None

        # Configure boto3 with longer timeout
        from botocore.config import Config
        config = Config(
            read_timeout=180,  # 3 minutes
            connect_timeout=10,
            retries={'max_attempts': 3}
        )

        # Check for Bedrock API key authentication
        api_key = os.environ.get('AWS_BEDROCK_API_KEY')
        if api_key:
            print("[INFO] Using Bedrock API key authentication")
            # Bedrock API keys use endpoint URL authentication
            self.bedrock_runtime = boto3.client(
                'bedrock-runtime',
                region_name=region_name,
                config=config,
                endpoint_url=f"https://bedrock-runtime.{region_name}.amazonaws.com"
            )
            # Store API key for custom auth
            self.api_key = api_key
        else:
            print("[INFO] Using IAM credentials for Bedrock authentication")
            self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region_name, config=config)
            self.api_key = None

        self.load_vocabulary()
        self.load_replacement_map()
    
    def load_vocabulary(self):
        """Load custom vocabulary data"""
        if not self.vocab_file.exists():
            print(f"WARNING:  Vocabulary file not found: {self.vocab_file}")
            print(f"   Script location: {Path(__file__).parent}")
            print(f"   Looking for: {self.vocab_file.absolute()}")
            print(f"   Run 'python build_vocabulary.py' first!")
            return
        
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            self.vocab_data = json.load(f)
        
        print(f"[OK] Loaded vocabulary:")
        print(f"  • {len(self.vocab_data['custom_vocabulary'])} custom terms")
        print(f"  • {self.vocab_data['metadata']['total_top_words']} common words")
        print(f"  • {self.vocab_data['metadata']['total_top_phrases']} common phrases")
    
    def load_replacement_map(self):
        """Load replacement map for reference"""
        if not self.replacement_map_file.exists():
            print(f"WARNING:  Replacement map not found: {self.replacement_map_file}")
            print(f"   Looking for: {self.replacement_map_file.absolute()}")
            return
        
        with open(self.replacement_map_file, 'r', encoding='utf-8') as f:
            self.replacement_map = json.load(f)
        
        print(f"[OK] Loaded {len(self.replacement_map)} known misspellings")
    
    def remove_filler_words(self, text):
        """
        Remove filler words (um, uh, ugh, etc.) from text.
        This is done AFTER paragraphing but BEFORE sending to Claude.
        """
        if not self.vocab_data or 'filter_words' not in self.vocab_data:
            return text
        
        result = text
        fillers_removed = 0
        
        for filler in self.vocab_data['filter_words']:
            # Remove standalone filler words (with word boundaries)
            pattern = r'\b' + re.escape(filler) + r'\b'
            before = result
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
            if result != before:
                fillers_removed += 1
        
        # Clean up extra spaces (but preserve paragraph breaks \n\n)
        # First, protect paragraph breaks
        result = result.replace('\n\n', '§PARA§')
        # Then clean up spaces
        result = re.sub(r'\s+', ' ', result).strip()
        result = re.sub(r'\s+([,.!?;:])', r'\1', result)
        # Restore paragraph breaks
        result = result.replace('§PARA§', '\n\n')
        
        if fillers_removed > 0:
            print(f"  [OK] Removed {fillers_removed} types of filler words")
        
        return result
    
    def build_vocabulary_context(self):
        """Build vocabulary context for Claude"""
        if not self.vocab_data:
            return ""
        
        context = []
        
        # Custom vocabulary (highest priority) - ALL of them
        context.append("=== CUSTOM VOCABULARY (Must be spelled correctly) ===")
        for entry in self.vocab_data['custom_vocabulary']:
            display = entry['display_as']
            sounds = entry.get('sounds_like', [])
            if sounds:
                context.append(f"- {display} (sounds like: {', '.join(sounds)})")
            else:
                context.append(f"- {display}")
        
        # ALL common phrases for full context
        context.append("\n=== COMMON PHRASES (Context for understanding) ===")
        phrases = [p['phrase'] for p in self.vocab_data['top_phrases']]  # ALL phrases
        context.append(", ".join(phrases))
        
        # ALL domain vocabulary for full context
        context.append("\n=== DOMAIN VOCABULARY (Context for understanding) ===")
        words = [w['word'] for w in self.vocab_data['top_words']]  # ALL words
        context.append(", ".join(words))
        
        return "\n".join(context)
    
    def build_known_errors_context(self):
        """Build known misspelling patterns for Claude"""
        if not self.replacement_map:
            return ""
        
        context = ["=== KNOWN COMMON ERRORS (Examples of what to fix) ==="]
        
        # Show ALL known errors from replacement map
        for incorrect, correct in self.replacement_map.items():
            context.append(f"- '{incorrect}' should be '{correct}'")
        
        return "\n".join(context)
    
    def build_claude_prompt(self, transcript_text):
        """Build optimized prompt - Claude returns only changes and paragraph positions, not full text"""
        vocab_context = self.build_vocabulary_context()
        errors_context = self.build_known_errors_context()
        
        prompt = f"""You are a meticulous transcript editor specializing in spiritual and philosophical content. Your task is to analyze this Ram Dass teaching transcript and identify:
1. Text corrections needed (vocabulary, grammar, transcription errors)
2. Where paragraph breaks (\\n\\n) should be inserted

Be EXTREMELY conservative with corrections - only fix obvious errors.

{vocab_context}

{errors_context}

=== CORRECTION RULES ===

DO FIX:
1. Clear misspellings of terms from the custom vocabulary above
   Examples: "lispensky" -> "Ouspensky", "maraj ji" -> "Maharaj-ji", "gurdjief" -> "Gurdjieff"
2. Obvious grammar errors (wrong verb tense, missing articles where clearly needed)
3. Duplicate words (e.g., "the the", "and and")
4. Clear transcription errors (gibberish, fragments missing words)

DO NOT:
1. Change words that are plausible and correct, even if uncommon
2. Add content that wasn't in the original
3. Rephrase or embellish the speaker's words
4. Change the speaker's grammatical style or voice
5. Fix intentional informal speech patterns
6. Change word order unless it's clearly an error
7. Invent words or phrases not in the transcript
8. Make changes you're not 95%+ confident about

=== PARAGRAPHING RULES ===

Identify where paragraph breaks (\\n\\n) should be inserted:
- Start a new paragraph when the speaker shifts to a new topic or example
- Start a new paragraph when there's a significant pause or change in narrative
- Start a new paragraph for each distinct story or anecdote
- Typical paragraph length: 3-8 sentences (but use your judgment based on content)
- Each paragraph should contain a complete thought or theme

Guidelines:
- Topic shifts (e.g., from talking about mothers to talking about death)
- New stories or examples (e.g., "When my mother...", "I have sat with my father...")
- Transitions between abstract concepts and concrete examples
- Natural speaking pauses where the speaker seems to "reset"

=== TRANSCRIPT ===

{transcript_text}

=== YOUR TASK ===

Analyze the transcript and return ONLY the changes needed. DO NOT return the full transcript.

Format as JSON:
{{
  "corrections": [
    {{
      "original": "exact text to find (10-30 chars for context)",
      "corrected": "corrected version",
      "reason": "brief explanation"
    }}
  ],
  "paragraph_breaks": [
    {{
      "after_text": "last 15-25 chars before break",
      "reason": "why break here (topic shift, new story, etc.)"
    }}
  ]
}}

IMPORTANT: 
- For corrections: Include enough context (10-30 chars) to uniquely identify the location
- For paragraph breaks: Provide the last 15-25 characters before where \\n\\n should be inserted
- Be precise - we'll use these to programmatically modify the original text
- Order paragraph_breaks from start to end of transcript

Remember: Be conservative with corrections, generous with paragraph breaks."""

        return prompt
    
    def apply_corrections_and_paragraphs(self, transcript_text, corrections, paragraph_breaks):
        """
        Apply corrections and paragraph breaks to transcript text.
        
        Args:
            transcript_text: Original transcript text
            corrections: List of {original, corrected, reason} dicts
            paragraph_breaks: List of {after_text, reason} dicts
            
        Returns:
            tuple: (refined_text, changes_applied_count)
        """
        result = transcript_text
        corrections_applied = 0
        corrections_failed = 0
        
        # Step 1: Apply text corrections
        print(f"\nApplying {len(corrections)} corrections...")
        for correction in corrections:
            original = correction.get('original', '')
            corrected = correction.get('corrected', '')
            reason = correction.get('reason', '')
            
            if original in result:
                result = result.replace(original, corrected, 1)
                corrections_applied += 1
                # Use ASCII-safe output to avoid encoding errors
                orig_safe = original[:40].encode('ascii', 'replace').decode('ascii')
                corr_safe = corrected[:40].encode('ascii', 'replace').decode('ascii')
                print(f"  [OK] {orig_safe}... -> {corr_safe}...")
            else:
                corrections_failed += 1
                orig_safe = original[:40].encode('ascii', 'replace').decode('ascii')
                print(f"  [X] Not found: {orig_safe}...")
        
        print(f"Corrections: {corrections_applied} applied, {corrections_failed} failed")
        
        # Step 2: Insert paragraph breaks
        print(f"\nInserting {len(paragraph_breaks)} paragraph breaks...")
        breaks_applied = 0
        breaks_failed = 0
        
        # Sort by position in text (process from end to start to preserve indices)
        sorted_breaks = []
        for pb in paragraph_breaks:
            after_text = pb.get('after_text', '')
            pos = result.rfind(after_text)  # Find last occurrence
            if pos != -1:
                sorted_breaks.append((pos + len(after_text), after_text, pb.get('reason', '')))
        
        # Sort in reverse order (end to start)
        sorted_breaks.sort(reverse=True)
        
        for pos, after_text, reason in sorted_breaks:
            # Check if there's already a paragraph break here
            if result[pos:pos+2] == '\n\n':
                # Use ASCII-safe output
                text_safe = after_text[-20:].encode('ascii', 'replace').decode('ascii')
                print(f"  [SKIP] Already has break after: ...{text_safe}...")
                continue
            
            # Strip any leading whitespace after the break position
            # to avoid " Right there..." indentation issues
            stripped_pos = pos
            while stripped_pos < len(result) and result[stripped_pos] in ' \t':
                stripped_pos += 1
            
            # Insert \n\n at position, removing any leading whitespace
            result = result[:pos] + '\n\n' + result[stripped_pos:]
            breaks_applied += 1
            # Use ASCII-safe output
            text_safe = after_text[-25:].encode('ascii', 'replace').decode('ascii')
            print(f"  [OK] Break after: ...{text_safe}...")
        
        breaks_failed = len(paragraph_breaks) - breaks_applied
        print(f"Paragraph breaks: {breaks_applied} inserted, {breaks_failed} skipped/failed")
        
        # Count final paragraphs
        paragraph_count = result.count('\n\n') + 1
        
        return result, {
            'corrections_applied': corrections_applied,
            'corrections_failed': corrections_failed,
            'breaks_applied': breaks_applied,
            'breaks_failed': breaks_failed,
            'paragraph_count': paragraph_count
        }
    
    def refine_transcript(self, transcript_text, verbose=False):
        """
        Use Claude to refine the transcript.
        
        Args:
            transcript_text: The transcript text to refine
            verbose: Print detailed progress
            
        Returns:
            dict with refined_transcript, changes, and summary
        """
        print(f"\n{'='*70}")
        print("REFINING TRANSCRIPT WITH CLAUDE OPUS 4.5")
        print(f"{'='*70}\n")
        
        print(f"Transcript length: {len(transcript_text):,} characters")
        
        # Build prompt
        prompt = self.build_claude_prompt(transcript_text)
        prompt_tokens = len(prompt) // 4  # Rough estimate
        print(f"Prompt length: {len(prompt):,} characters (~{prompt_tokens:,} tokens)")
        
        if prompt_tokens > 100000:
            print("WARNING:  Warning: Prompt may exceed 100k token context window")
            print("          Consider processing in smaller chunks if errors occur")
        
        print("\nSending to Claude Sonnet 4.5...")
        if verbose:
            print("  [VERBOSE] Extended thinking enabled - Claude will show its reasoning process")
            print("  [VERBOSE] Streaming enabled - showing real-time progress")
        
        # Retry logic for modelStreamErrorException
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    print(f"\n⚠️  Retry attempt {attempt}/{max_retries} after {retry_delay}s delay...")
                    time.sleep(retry_delay)
                
                result = self._call_claude_with_streaming(prompt, transcript_text, verbose)
                return result  # Success - return immediately
                
            except ModelStreamError as e:
                if attempt < max_retries:
                    print(f"\n⚠️  ModelStreamError: {e}")
                    print(f"   Waiting {retry_delay}s before retry {attempt + 1}/{max_retries}...")
                    continue  # Try again
                else:
                    # Final attempt failed
                    print(f"\n❌ All {max_retries} retry attempts failed")
                    raise
            except Exception as e:
                # Non-retryable error - fail immediately
                print(f"\n[ERROR] Error calling Claude: {e}")
                import traceback
                traceback.print_exc()
                raise
    
    def _call_claude_with_streaming(self, prompt, transcript_text, verbose):
        """Internal method to call Claude with streaming (for retry logic)."""
        try:
            # Build request parameters
            request_params = {
                'modelId': 'anthropic.claude-opus-4-5-20251101-v1:0',
                'messages': [
                    {
                        'role': 'user',
                        'content': [{'text': prompt}]
                    }
                ],
                'inferenceConfig': {
                    'maxTokens': 25000,  # Allow full 25k output tokens for complete transcripts
                    'temperature': 1.0 if verbose else 0.3  # Must be 1.0 when thinking is enabled
                }
            }
            
            # Enable extended thinking in verbose mode
            if verbose:
                request_params['additionalModelRequestFields'] = {
                    'thinking': {
                        'type': 'enabled',
                        'budget_tokens': 5000  # Allow up to 5k tokens for thinking
                    }
                }
            
            # Use streaming API
            response_stream = self.bedrock_runtime.converse_stream(**request_params)
            
            # Collect streamed content
            thinking_text = None
            result_text = ""
            thinking_shown = False
            input_tokens = 0
            output_tokens = 0
            last_event_time = None
            
            print()  # New line before streaming output
            
            # Process event stream
            import time
            for event in response_stream['stream']:
                last_event_time = time.time()
                
                if 'contentBlockStart' in event:
                    block_start = event['contentBlockStart']
                    if 'start' in block_start and 'thinking' in block_start['start']:
                        if verbose:
                            print(f"\n{'='*70}")
                            print("CLAUDE'S THINKING PROCESS:")
                            print(f"{'='*70}")
                            thinking_shown = True
                
                elif 'contentBlockDelta' in event:
                    delta = event['contentBlockDelta']['delta']
                    
                    if 'thinking' in delta:
                        # Thinking content
                        thinking_chunk = delta['thinking']
                        if thinking_text is None:
                            thinking_text = thinking_chunk
                        else:
                            thinking_text += thinking_chunk
                        
                        if verbose:
                            print(thinking_chunk, end='', flush=True)
                    
                    elif 'text' in delta:
                        # Result text content
                        text_chunk = delta['text']
                        result_text += text_chunk
                        
                        if verbose:
                            if thinking_shown:
                                print(f"\n{'='*70}\n")
                                print("REFINED TRANSCRIPT:")
                                print(f"{'='*70}\n")
                                thinking_shown = False
                            print(text_chunk, end='', flush=True)
                
                elif 'metadata' in event:
                    usage = event['metadata'].get('usage', {})
                    input_tokens = usage.get('inputTokens', 0)
                    output_tokens = usage.get('outputTokens', 0)
                
                elif 'internalServerException' in event:
                    error_msg = event['internalServerException'].get('message', 'Unknown server error')
                    raise Exception(f"AWS Bedrock server error: {error_msg}")
                
                elif 'modelStreamErrorException' in event:
                    error_msg = event['modelStreamErrorException'].get('message', 'Unknown streaming error')
                    # Mark as retryable error
                    raise ModelStreamError(f"Model streaming error: {error_msg}")
                
                elif 'throttlingException' in event:
                    error_msg = event['throttlingException'].get('message', 'Request throttled')
                    raise Exception(f"Throttling error: {error_msg}")
                
                elif 'validationException' in event:
                    error_msg = event['validationException'].get('message', 'Validation failed')
                    raise Exception(f"Validation error: {error_msg}")
            
            print()  # New line after streaming
            
            # Verify we got content
            if not result_text:
                raise Exception("No text content received from Claude stream")
            
            if verbose:
                stream_duration = time.time() - last_event_time if last_event_time else 0
                print(f"\n[OK] Claude response complete")
                print(f"  • Input tokens: {input_tokens:,}")
                print(f"  • Output tokens: {output_tokens:,}")
                print(f"  • Estimated cost: ${(input_tokens * 0.003 / 1000 + output_tokens * 0.015 / 1000):.4f}")
                if stream_duration > 0:
                    print(f"  • Stream completed in {stream_duration:.1f}s")
            
            # Extract JSON from response (handle markdown code blocks)
            json_text = result_text.strip()
            if json_text.startswith('```'):
                # Extract JSON from code block
                lines = json_text.split('\n')
                json_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else json_text
                json_text = json_text.replace('```json', '').replace('```', '').strip()
            
            # If JSON still has extra text, try to extract just the JSON object
            if not json_text.startswith('{'):
                start = json_text.find('{')
                end = json_text.rfind('}')
                if start != -1 and end != -1:
                    json_text = json_text[start:end+1]
            
            result = json.loads(json_text)
            
            # Get corrections and paragraph breaks from Claude
            corrections = result.get('corrections', [])
            paragraph_breaks = result.get('paragraph_breaks', [])
            
            print(f"\n{'='*70}")
            print(f"Claude identified {len(corrections)} corrections")
            print(f"Claude identified {len(paragraph_breaks)} paragraph breaks")
            print(f"{'='*70}")
            
            # Apply corrections and paragraph breaks locally
            refined_text, apply_stats = self.apply_corrections_and_paragraphs(
                transcript_text, 
                corrections, 
                paragraph_breaks
            )
            
            # Show summary
            print(f"\n{'='*70}")
            print(f"REFINEMENT COMPLETE")
            print(f"  • Corrections applied: {apply_stats['corrections_applied']}/{len(corrections)}")
            print(f"  • Paragraph breaks inserted: {apply_stats['breaks_applied']}/{len(paragraph_breaks)}")
            print(f"  • Final paragraph count: {apply_stats['paragraph_count']}")
            print(f"{'='*70}")
            
            # Return in expected format
            return {
                'refined_transcript': refined_text,
                'changes': corrections,
                'summary': {
                    'total_changes': apply_stats['corrections_applied'],
                    'paragraph_count': apply_stats['paragraph_count'],
                    'corrections_failed': apply_stats['corrections_failed'],
                    'breaks_applied': apply_stats['breaks_applied']
                }
            }
            
        except ModelStreamError:
            # Re-raise ModelStreamError for retry logic
            raise
        except Exception as e:
            # Other errors are not retryable
            raise
    
    def process_file(self, input_file, output_file=None, verbose=False):
        """
        Process a transcript file.
        
        Args:
            input_file: Path to input transcript file (.txt)
            output_file: Path to output file (default: input_refined.txt)
            verbose: Print detailed progress
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Read input
        print(f"\nReading: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
        
        # Remove filler words BEFORE sending to Claude
        print("\nRemoving filler words (um, uh, ugh, etc.)...")
        transcript_text = self.remove_filler_words(transcript_text)
        
        # Refine
        result = self.refine_transcript(transcript_text, verbose=verbose)
        
        # Determine output path
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = input_path.parent / f"{input_path.stem}_refined.txt"
        
        # Save refined transcript
        refined_text = result.get('refined_transcript', transcript_text)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(refined_text)
        print(f"\n[OK] Saved refined transcript: {output_path}")
        
        # Save changes log
        changes_path = output_path.parent / f"{output_path.stem}_changes.json"
        with open(changes_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved changes log: {changes_path}")
        
        return output_path, changes_path


def refine_transcript_text(transcript_text: str, output_path: str, verbose: bool = False, region_name: str = "us-east-1") -> dict:
    """
    Refine transcript text using Claude and save results.

    Designed for worker pools - handles full refinement workflow.

    Args:
        transcript_text: Raw transcript text to refine
        output_path: Path for output file (will add _refined.txt)
        verbose: Enable detailed output
        region_name: AWS region for Bedrock

    Returns:
        dict with 'refined_path', 'changes_path', 'summary'
    """
    refiner = ClaudeTranscriptRefiner(region_name=region_name)

    # Remove filler words first
    print("\nRemoving filler words...")
    cleaned_text = refiner.remove_filler_words(transcript_text)

    # Refine with Claude
    result = refiner.refine_transcript(cleaned_text, verbose=verbose)

    # Save outputs
    output_path = Path(output_path)
    refined_path = output_path.parent / f"{output_path.stem}_refined.txt"
    changes_path = output_path.parent / f"{output_path.stem}_refined_changes.json"

    refined_text = result.get('refined_transcript', cleaned_text)

    try:
        with open(refined_path, 'w', encoding='utf-8') as f:
            f.write(refined_text)
        print(f"[OK] Saved refined transcript: {refined_path}")
    except IOError as e:
        raise IOError(f"Failed to write refined transcript to {refined_path}: {e}")

    try:
        with open(changes_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved changes log: {changes_path}")
    except IOError as e:
        raise IOError(f"Failed to write changes log to {changes_path}: {e}")

    return {
        'refined_path': str(refined_path),
        'changes_path': str(changes_path),
        'summary': result.get('summary', {})
    }


def main():
    parser = argparse.ArgumentParser(
        description='Refine transcript using Claude Sonnet 4.5 with vocabulary awareness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Refine a formatted transcript (verbose by default)
  python claude_refine_transcript.py "downloads/Clip 1_1969_formatted.txt"
  
  # Specify output file
  python claude_refine_transcript.py input.txt --output refined.txt
  
  # Silent mode (no streaming output)
  python claude_refine_transcript.py input.txt --silent
        """
    )
    parser.add_argument('input_file', help='Path to transcript file (.txt)')
    parser.add_argument('-o', '--output', help='Output file path (default: input_refined.txt)')
    parser.add_argument('-s', '--silent', action='store_true', 
                        help='Silent mode - disable streaming output and thinking display')
    parser.add_argument('--region', default='us-east-1',
                        help='AWS region (default: us-east-1)')
    
    args = parser.parse_args()
    
    # Initialize refiner
    refiner = ClaudeTranscriptRefiner(region_name=args.region)
    
    # Verbose is default (ON unless --silent is specified)
    verbose = not args.silent
    
    # Process file
    try:
        refiner.process_file(args.input_file, args.output, verbose)
        print(f"\n{'='*70}")
        print("[OK] REFINEMENT COMPLETE")
        print(f"{'='*70}\n")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
