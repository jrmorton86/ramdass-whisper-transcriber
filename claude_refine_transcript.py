#!/usr/bin/env python3
"""
Claude-powered transcript refinement using AWS Bedrock.

Uses Claude Sonnet 4.5 with the same vocabulary context given to Whisper
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
import boto3
from pathlib import Path
import argparse


class ClaudeTranscriptRefiner:
    def __init__(self, 
                 vocab_file="keyword_lists/whisper_vocabulary.json",
                 replacement_map_file="keyword_lists/replacement_map.json",
                 region_name="us-east-1"):
        self.vocab_file = Path(vocab_file)
        self.replacement_map_file = Path(replacement_map_file)
        self.vocab_data = None
        self.replacement_map = None
        
        # Configure boto3 with longer timeout
        from botocore.config import Config
        config = Config(
            read_timeout=180,  # 3 minutes
            connect_timeout=10,
            retries={'max_attempts': 3}
        )
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region_name, config=config)
        
        self.load_vocabulary()
        self.load_replacement_map()
    
    def load_vocabulary(self):
        """Load custom vocabulary data"""
        if not self.vocab_file.exists():
            print(f"WARNING:  Vocabulary file not found: {self.vocab_file}")
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
        """Build the complete prompt for Claude - returns full refined transcript with paragraphs"""
        vocab_context = self.build_vocabulary_context()
        errors_context = self.build_known_errors_context()
        
        prompt = f"""You are a meticulous transcript editor specializing in spiritual and philosophical content. Your task is to refine this Ram Dass teaching transcript by:
1. Fixing clear errors (vocabulary, grammar, transcription mistakes)
2. Adding proper paragraph breaks to make it readable

Be EXTREMELY conservative with corrections - only fix obvious errors.

{vocab_context}

{errors_context}

=== CORRECTION RULES ===

DO FIX:
1. Clear misspellings of terms from the custom vocabulary above
   Examples: "lispensky" → "Ouspensky", "maraj ji" → "Maharaj-ji", "gurdjief" → "Gurdjieff"
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

Add paragraph breaks (double newlines: \\n\\n) to make the text readable:
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

Return the refined transcript with:
1. All clear errors corrected
2. Proper paragraph breaks added using \\n\\n
3. The speaker's authentic voice preserved

Format as JSON:
{{
  "refined_transcript": "full transcript with corrections and \\n\\n paragraph breaks",
  "changes_made": [
    {{
      "type": "correction",
      "original": "text before",
      "corrected": "text after",
      "reason": "explanation"
    }}
  ],
  "paragraph_count": number_of_paragraphs_created
}}

Remember: Be conservative with corrections, generous with paragraph breaks. Make it readable while preserving the speaker's authentic voice."""

        return prompt
    
    def apply_corrections(self, transcript_text, corrections):
        """Apply corrections to transcript text"""
        result = transcript_text
        changes_made = []
        
        for correction in corrections:
            original = correction['original_text']
            corrected = correction['corrected_text']
            
            # Check if original text exists in transcript
            if original in result:
                # Replace first occurrence
                result = result.replace(original, corrected, 1)
                changes_made.append(correction)
                print(f"  [OK] Applied: {original[:50]}... -> {corrected[:50]}...")
            else:
                print(f"  [WARNING] Could not find: {original[:50]}...")
        
        return result, changes_made
    
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
        print("REFINING TRANSCRIPT WITH CLAUDE SONNET 4.5")
        print(f"{'='*70}\n")
        
        print(f"Transcript length: {len(transcript_text):,} characters")
        
        # Build prompt
        prompt = self.build_claude_prompt(transcript_text)
        prompt_tokens = len(prompt) // 4  # Rough estimate
        print(f"Prompt length: {len(prompt):,} characters (~{prompt_tokens:,} tokens)")
        
        if prompt_tokens > 100000:
            print("WARNING:  Warning: Prompt may exceed 100k token context window")
        
        print("\nSending to Claude Sonnet 4.5...")
        
        # Call Claude via Bedrock
        try:
            response = self.bedrock_runtime.converse(
                modelId='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                messages=[
                    {
                        'role': 'user',
                        'content': [{'text': prompt}]
                    }
                ],
                inferenceConfig={
                    'maxTokens': 20000,  # Increased since we're returning full transcript
                    'temperature': 0.3  # Low temperature for conservative edits
                }
            )
            
            # Parse response
            result_text = response['output']['message']['content'][0]['text']
            
            if verbose:
                usage = response.get('usage', {})
                input_tokens = usage.get('inputTokens', 0)
                output_tokens = usage.get('outputTokens', 0)
                print(f"\n[OK] Claude response received")
                print(f"  • Input tokens: {input_tokens:,}")
                print(f"  • Output tokens: {output_tokens:,}")
                print(f"  • Estimated cost: ${(input_tokens * 0.003 / 1000 + output_tokens * 0.015 / 1000):.4f}")
            
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
            
            # Get refined transcript and changes
            refined_text = result.get('refined_transcript', transcript_text)
            changes_made = result.get('changes_made', [])
            paragraph_count = result.get('paragraph_count', 0)
            
            print(f"\n{'='*70}")
            print(f"Claude made {len(changes_made)} corrections")
            print(f"Created {paragraph_count} paragraphs")
            print(f"{'='*70}")
            
            if changes_made:
                print("\nChanges made:")
                for change in changes_made[:10]:  # Show first 10
                    print(f"  • {change.get('original', '')[:50]}... -> {change.get('corrected', '')[:50]}...")
                    print(f"    Reason: {change.get('reason', 'N/A')}")
                if len(changes_made) > 10:
                    print(f"  ... and {len(changes_made) - 10} more changes")
            
            # Return in expected format
            return {
                'refined_transcript': refined_text,
                'changes': changes_made,
                'summary': {
                    'total_changes': len(changes_made),
                    'paragraph_count': paragraph_count
                }
            }
            
        except Exception as e:
            print(f"\n[ERROR] Error calling Claude: {e}")
            import traceback
            traceback.print_exc()
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


def main():
    parser = argparse.ArgumentParser(
        description='Refine transcript using Claude Sonnet 4.5 with vocabulary awareness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Refine a formatted transcript
  python claude_refine_transcript.py "downloads/Clip 1_1969_formatted.txt"
  
  # Specify output file
  python claude_refine_transcript.py input.txt --output refined.txt
  
  # Verbose mode (show Claude's full response)
  python claude_refine_transcript.py input.txt --verbose
        """
    )
    parser.add_argument('input_file', help='Path to transcript file (.txt)')
    parser.add_argument('-o', '--output', help='Output file path (default: input_refined.txt)')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Show detailed progress and Claude responses')
    parser.add_argument('--region', default='us-east-1',
                        help='AWS region (default: us-east-1)')
    
    args = parser.parse_args()
    
    # Initialize refiner
    refiner = ClaudeTranscriptRefiner(region_name=args.region)
    
    # Process file
    try:
        refiner.process_file(args.input_file, args.output, args.verbose)
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
