#!/usr/bin/env python3
"""
Enhanced Whisper transcription with custom vocabulary support.

Uses the vocabulary built from AWS Transcribe exports and keyword analysis
to improve transcription accuracy for Ram Dass content.
"""

import whisper
import json
import re
from pathlib import Path


class VocabularyEnhancedTranscriber:
    def __init__(self, vocab_file="keyword_lists/whisper_vocabulary.json",
                 replacement_map_file="keyword_lists/replacement_map.json"):
        self.vocab_file = Path(vocab_file)
        self.replacement_map_file = Path(replacement_map_file)
        self.vocab_data = None
        self.replacement_map = None
        self.initial_prompt = None
        
        self.load_vocabulary()
        self.load_replacement_map()
        self.build_initial_prompt()
    
    def load_vocabulary(self):
        """Load custom vocabulary data"""
        if not self.vocab_file.exists():
            print(f"WARNING: Vocabulary file not found: {self.vocab_file}")
            print("   Run 'python build_vocabulary.py' first!")
            return
        
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            self.vocab_data = json.load(f)
        
        print(f"[OK] Loaded vocabulary:")
        print(f"  - {len(self.vocab_data['custom_vocabulary'])} custom terms")
        print(f"  - {self.vocab_data['metadata']['total_top_words']} common words")
        print(f"  - {self.vocab_data['metadata']['total_top_phrases']} common phrases")
    
    def load_replacement_map(self):
        """Load post-processing replacement map"""
        if not self.replacement_map_file.exists():
            print(f"WARNING: Replacement map not found: {self.replacement_map_file}")
            return
        
        with open(self.replacement_map_file, 'r', encoding='utf-8') as f:
            self.replacement_map = json.load(f)
        
        print(f"[OK] Loaded {len(self.replacement_map)} replacement rules")
    
    def build_initial_prompt(self, max_tokens=32768):
        """
        Build initial prompt for Whisper to prime it with vocabulary.
        
        Whisper uses initial_prompt to bias transcription toward specific terms.
        Target ~32,768 tokens (roughly 131,000 characters) for maximum context.
        """
        if not self.vocab_data:
            return
        
        # Target character count (rough estimate: 1 token ≈ 4 chars)
        target_chars = max_tokens * 4
        
        full_prompt = ""
        
        # Section 1: Custom vocabulary (highest priority) - ALWAYS include
        custom_terms = [entry['display_as'] for entry in self.vocab_data['custom_vocabulary']]
        full_prompt += "Key spiritual terms: " + ", ".join(custom_terms) + ". "
        
        # Section 2: ALL common phrases (use everything we have!)
        if len(full_prompt) < target_chars * 0.8:  # Leave room for more
            all_phrases = [p['phrase'] for p in self.vocab_data['top_phrases']]
            full_prompt += "Common topics and phrases: " + ", ".join(all_phrases) + ". "
        
        # Section 3: ALL domain vocabulary
        if len(full_prompt) < target_chars * 0.8:
            exclude_common = {'first', 'years', 'people', 'thing', 'new', 'time', 'way', 
                             'day', 'minutes', 'ago', 'last', 'second', 'next', 'days'}
            all_words = [w['word'] for w in self.vocab_data['top_words'] 
                        if w['word'].lower() not in exclude_common]
            full_prompt += "Vocabulary context: " + ", ".join(all_words) + ". "
        
        # Section 4: Repeat important terms to fill remaining space
        if len(full_prompt) < target_chars:
            # Add extended context sentences multiple times
            context_base = [
                "This is a spiritual teaching by Ram Dass, formerly Richard Alpert, discussing consciousness, meditation, and awareness. ",
                "Maharaj-ji, also known as Neem Karoli Baba, is the primary spiritual teacher referenced. ",
                "Topics include dharma, karma, bhakti yoga, meditation practices, and Eastern philosophy. ",
                "Locations mentioned: India, Vrindavan, Rishikesh, America, Harvard, Himalayas. ",
                "Practices: meditation, pranayama, kirtan, satsang, seva, japa, mantra repetition. ",
                "Key concepts: consciousness, awareness, presence, being, mind, ego, soul, spirit. ",
                "The teachings emphasize love, service, compassion, and spiritual awakening. ",
                "Discussions of psychedelics, LSD, Timothy Leary, and consciousness exploration. ",
                "References to Hinduism, Buddhism, yoga, tantra, and mystical traditions. ",
                "Mentions of gurus, saints, and spiritual masters from various traditions. ",
                "Common phrases: somebody or other, something or other. ",
            ]
            
            # Keep adding sentences until we reach target
            while len(full_prompt) < target_chars * 0.95:
                for sentence in context_base:
                    full_prompt += sentence
                    if len(full_prompt) >= target_chars * 0.95:
                        break
        
        self.initial_prompt = full_prompt.strip()
        
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        estimated_tokens = len(self.initial_prompt) // 4
        
        print(f"\n{'='*70}")
        print(f"INITIAL PROMPT FOR WHISPER")
        print(f"{'='*70}")
        print(f"Length: {len(self.initial_prompt):,} characters")
        print(f"Estimated tokens: ~{estimated_tokens:,} / {max_tokens:,} target")
        print(f"Coverage: {(estimated_tokens/max_tokens)*100:.1f}%")
        print(f"\n{'='*70}")
        print(f"FULL PROMPT CONTENT:")
        print(f"{'='*70}\n")
        print(self.initial_prompt)
        print(f"\n{'='*70}")
        print(f"END OF PROMPT")
        print(f"{'='*70}\n")
    
    def apply_replacements(self, text):
        """
        Apply vocabulary-based replacements to transcribed text.
        
        This fixes common transcription errors for specialized terms.
        """
        if not self.replacement_map:
            return text
        
        result = text
        replacements_made = 0
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_replacements = sorted(
            self.replacement_map.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        for incorrect, correct in sorted_replacements:
            # Case-insensitive replacement with word boundaries
            pattern = r'\b' + re.escape(incorrect) + r'\b'
            before = result
            result = re.sub(pattern, correct, result, flags=re.IGNORECASE)
            if result != before:
                replacements_made += 1
        
        if replacements_made > 0:
            print(f"  [OK] Applied {replacements_made} vocabulary corrections")
        
        return result
    
    def filter_filler_words(self, text):
        """Remove filler words based on filter list"""
        if not self.vocab_data or 'filter_words' not in self.vocab_data:
            return text
        
        result = text
        for filler in self.vocab_data['filter_words']:
            # Remove standalone filler words (with word boundaries)
            pattern = r'\b' + re.escape(filler) + r'\b'
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def transcribe(self, audio_path, model_name="base", language="en",
                   apply_vocab_corrections=True, remove_fillers=False,
                   device=None, **whisper_kwargs):
        """
        Transcribe audio with vocabulary enhancement.
        
        Args:
            audio_path: Path to audio file
            model_name: Whisper model size (tiny, base, small, medium, large)
            language: Language code (default: en)
            apply_vocab_corrections: Apply post-processing corrections
            remove_fillers: Remove filler words (uh, um, etc.)
            device: CUDA device to use (e.g., 'cuda:0', 'cuda:1')
            **whisper_kwargs: Additional arguments for whisper.transcribe()
        
        Returns:
            dict with 'text', 'segments', and metadata
        """
        print(f"\n{'='*70}")
        print(f"TRANSCRIBING: {audio_path}")
        print(f"{'='*70}\n")
        
        # Load model - prioritize GPU VRAM over system RAM
        print(f"Loading Whisper model: {model_name}")
        
        # Import torch for device detection
        import torch
        
        # Determine device: use specified device, or auto-detect CUDA
        if device:
            target_device = device
            print(f"Using specified device: {device}")
        else:
            # Auto-detect: use CUDA if available, otherwise CPU
            if torch.cuda.is_available():
                target_device = "cuda"
                print(f"Auto-detected CUDA - Loading directly to GPU VRAM")
                print(f"   GPU 0: {torch.cuda.get_device_name(0)}")
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"   Total VRAM: {vram_gb:.1f} GB")
            else:
                target_device = "cpu"
                print(f"CUDA not available - Using CPU")
        
        # Load model with explicit device to ensure VRAM usage
        model = whisper.load_model(model_name, device=target_device)
        
        # Verify model is on correct device and check precision
        if target_device.startswith('cuda'):
            actual_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            print(f"[OK] Model loaded on: {actual_device}")
            print(f"   Model precision: {model_dtype}")
            
            # Show VRAM usage
            vram_allocated = torch.cuda.memory_allocated(actual_device.index or 0) / 1024**3
            print(f"   VRAM allocated: {vram_allocated:.2f} GB")
        else:
            print(f"[OK] Model loaded on: {target_device}")
        
        # Prepare transcription options
        options = {
            'language': language,
            'initial_prompt': self.initial_prompt,
            'condition_on_previous_text': False,  # Prevent repetition hallucinations
            'compression_ratio_threshold': 2.4,   # Detect repetitive output
            'logprob_threshold': -1.0,            # Filter low-confidence segments
            'no_speech_threshold': 0.6,           # Skip silence better
            'verbose': False,  # Disable to prevent Unicode encoding errors on Windows console
            **whisper_kwargs
        }
        
        # CRITICAL: Force FP16 on GPU for optimal VRAM usage (50% reduction)
        if target_device.startswith('cuda'):
            # Override any fp16 setting in whisper_kwargs to ensure it's enabled
            options['fp16'] = True
            
            print(f"\n{'='*70}")
            print("TRANSCRIPTION SETTINGS")
            print(f"{'='*70}")
            print(f"[OK] FP16 precision: ENABLED (forced)")
            print(f"   Expected VRAM: ~2.5 GB for medium model")
            print(f"   Performance: 2-3x faster than CPU")
            print(f"   Memory savings: 50% vs FP32")
            print(f"\nVocabulary-enhanced prompt: ENABLED")
            print("Anti-hallucination settings: ENABLED (condition_on_previous_text=False)")
            print(f"{'='*70}")
        else:
            options['fp16'] = False
            print(f"\n{'='*70}")
            print("TRANSCRIPTION SETTINGS")
            print(f"{'='*70}")
            print(f"⚠️  CPU Mode: FP16 disabled (CPU doesn't support FP16)")
            print(f"\nVocabulary-enhanced prompt: ENABLED")
            print("Anti-hallucination settings: ENABLED (condition_on_previous_text=False)")
            print(f"{'='*70}")
        
        # Show memory usage tracking for GPU
        if target_device.startswith('cuda'):
            device_idx = actual_device.index if actual_device.index is not None else 0
            vram_before = torch.cuda.memory_allocated(device_idx) / 1024**3
            print(f"\n[VRAM] Before transcription: {vram_before:.2f} GB")
        
        # Transcribe (NOTE: Whisper architecture requires audio in CPU RAM first)
        # Audio preprocessing (load, resample) happens on CPU
        # Model inference happens on GPU with FP16
        result = model.transcribe(str(audio_path), **options)
        
        # Show final memory usage
        if target_device.startswith('cuda'):
            vram_after = torch.cuda.memory_allocated(device_idx) / 1024**3
            print(f"[VRAM] After transcription: {vram_after:.2f} GB")
            print(f"   Peak usage during transcription: ~{vram_after:.2f} GB")
        
        # Post-process text
        original_text = result['text']
        processed_text = original_text
        
        print(f"\n{'='*70}")
        print("POST-PROCESSING")
        print(f"{'='*70}")
        
        if apply_vocab_corrections:
            print("\nApplying vocabulary corrections...")
            processed_text = self.apply_replacements(processed_text)
        
        if remove_fillers:
            print("\nRemoving filler words...")
            processed_text = self.filter_filler_words(processed_text)
        
        # Update result
        result['text'] = processed_text
        result['original_text'] = original_text
        result['vocab_enhanced'] = apply_vocab_corrections
        result['fillers_removed'] = remove_fillers
        
        print(f"\n{'='*70}")
        print("TRANSCRIPTION COMPLETE")
        print(f"{'='*70}")
        print(f"\nLength: {len(processed_text)} characters")
        
        return result
    
    def save_transcription(self, result, output_path):
        """Save transcription to file"""
        output_path = Path(output_path)
        
        # Save text file (append .txt to preserve full filename)
        text_path = Path(str(output_path) + '.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        print(f"[OK] Saved text: {text_path}")
        
        # Save JSON with full details (append .json to preserve full filename)
        json_path = Path(str(output_path) + '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved JSON: {json_path}")
        
        return text_path, json_path


def transcribe_audio(audio_path, model, vocab_data=None, replacement_map=None,
                     apply_vocab_corrections=True, remove_fillers=False, device=None):
    """
    Transcribe audio with a pre-loaded Whisper model.

    This function is designed for worker pools where the model is loaded once
    and reused for multiple transcriptions.

    Args:
        audio_path: Path to audio file
        model: Pre-loaded Whisper model
        vocab_data: Vocabulary data dict (optional, for corrections)
        replacement_map: Replacement map dict (optional, for corrections)
        apply_vocab_corrections: Apply post-processing corrections
        remove_fillers: Remove filler words
        device: CUDA device string (for logging only - model already on device)

    Returns:
        dict with 'text', 'segments', and metadata
    """
    import torch

    print(f"\n{'='*70}")
    print(f"TRANSCRIBING: {audio_path}")
    print(f"{'='*70}\n")

    # Build initial prompt from vocabulary if available
    initial_prompt = None
    if vocab_data:
        initial_prompt = _build_initial_prompt(vocab_data)

    # Detect device from model
    target_device = str(next(model.parameters()).device)
    is_cuda = target_device.startswith('cuda')

    # Prepare transcription options
    options = {
        'language': 'en',
        'initial_prompt': initial_prompt,
        'condition_on_previous_text': False,
        'compression_ratio_threshold': 2.4,
        'logprob_threshold': -1.0,
        'no_speech_threshold': 0.6,
        'verbose': False,
        'fp16': is_cuda
    }

    if is_cuda:
        device_idx = int(target_device.split(':')[1]) if ':' in target_device else 0
        vram_before = torch.cuda.memory_allocated(device_idx) / 1024**3
        print(f"[VRAM] Before transcription: {vram_before:.2f} GB")

    # Transcribe
    result = model.transcribe(str(audio_path), **options)

    if is_cuda:
        vram_after = torch.cuda.memory_allocated(device_idx) / 1024**3
        print(f"[VRAM] After transcription: {vram_after:.2f} GB")

    # Post-process text
    original_text = result['text']
    processed_text = original_text

    if apply_vocab_corrections and replacement_map:
        processed_text = _apply_replacements(processed_text, replacement_map)

    if remove_fillers and vocab_data and 'filter_words' in vocab_data:
        processed_text = _filter_filler_words(processed_text, vocab_data['filter_words'])

    result['text'] = processed_text
    result['original_text'] = original_text
    result['vocab_enhanced'] = apply_vocab_corrections
    result['fillers_removed'] = remove_fillers

    print(f"\n{'='*70}")
    print(f"TRANSCRIPTION COMPLETE - {len(processed_text)} characters")
    print(f"{'='*70}")

    return result


def _build_initial_prompt(vocab_data, max_tokens=32768):
    """Build initial prompt for Whisper from vocabulary data."""
    target_chars = max_tokens * 4
    full_prompt = ""

    custom_terms = [entry['display_as'] for entry in vocab_data.get('custom_vocabulary', [])]
    full_prompt += "Key spiritual terms: " + ", ".join(custom_terms) + ". "

    if len(full_prompt) < target_chars * 0.8:
        all_phrases = [p['phrase'] for p in vocab_data.get('top_phrases', [])]
        full_prompt += "Common topics and phrases: " + ", ".join(all_phrases) + ". "

    if len(full_prompt) < target_chars * 0.8:
        exclude_common = {'first', 'years', 'people', 'thing', 'new', 'time', 'way',
                         'day', 'minutes', 'ago', 'last', 'second', 'next', 'days'}
        all_words = [w['word'] for w in vocab_data.get('top_words', [])
                    if w['word'].lower() not in exclude_common]
        full_prompt += "Vocabulary context: " + ", ".join(all_words) + ". "

    return full_prompt.strip()


def _apply_replacements(text, replacement_map):
    """Apply vocabulary-based replacements to transcribed text."""
    import re
    result = text
    sorted_replacements = sorted(replacement_map.items(), key=lambda x: len(x[0]), reverse=True)

    for incorrect, correct in sorted_replacements:
        pattern = r'\b' + re.escape(incorrect) + r'\b'
        result = re.sub(pattern, correct, result, flags=re.IGNORECASE)

    return result


def _filter_filler_words(text, filter_words):
    """Remove filler words from text."""
    import re
    result = text
    for filler in filter_words:
        pattern = r'\b' + re.escape(filler) + r'\b'
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def load_vocabulary_data(vocab_file="keyword_lists/whisper_vocabulary.json",
                         replacement_map_file="keyword_lists/replacement_map.json"):
    """
    Load vocabulary data and replacement map.

    Args:
        vocab_file: Path to vocabulary JSON
        replacement_map_file: Path to replacement map JSON

    Returns:
        tuple: (vocab_data, replacement_map) - either can be None if file not found
    """
    import json
    from pathlib import Path

    script_dir = Path(__file__).parent

    vocab_data = None
    vocab_path = script_dir / vocab_file
    if vocab_path.exists():
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        print(f"[OK] Loaded vocabulary: {len(vocab_data.get('custom_vocabulary', []))} terms")

    replacement_map = None
    replacement_path = script_dir / replacement_map_file
    if replacement_path.exists():
        with open(replacement_path, 'r', encoding='utf-8') as f:
            replacement_map = json.load(f)
        print(f"[OK] Loaded {len(replacement_map)} replacement rules")

    return vocab_data, replacement_map


def save_transcription_result(result, output_path):
    """
    Save transcription result to JSON and TXT files.

    Args:
        result: Whisper transcription result dict
        output_path: Base output path (without extension)

    Returns:
        tuple: (txt_path, json_path)
    """
    import json
    from pathlib import Path

    output_path = Path(output_path)

    txt_path = Path(str(output_path) + '.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(result['text'])

    json_path = Path(str(output_path) + '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return txt_path, json_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Transcribe audio with custom vocabulary enhancement'
    )
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('-m', '--model', default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size (default: base)')
    parser.add_argument('-o', '--output', help='Output file path (without extension)')
    parser.add_argument('-d', '--device', help='CUDA device to use (e.g., cuda:0, cuda:1)')
    parser.add_argument('--no-corrections', action='store_true',
                        help='Disable vocabulary corrections')
    parser.add_argument('--remove-fillers', action='store_true',
                        help='Remove filler words (uh, um, etc.)')
    parser.add_argument('--language', default='en',
                        help='Language code (default: en)')
    
    args = parser.parse_args()
    
    # Initialize transcriber
    transcriber = VocabularyEnhancedTranscriber()
    
    # Transcribe
    result = transcriber.transcribe(
        args.audio_file,
        model_name=args.model,
        language=args.language,
        apply_vocab_corrections=not args.no_corrections,
        remove_fillers=args.remove_fillers,
        device=args.device
    )
    
    # Save output
    if args.output:
        output_path = args.output
    else:
        # Default: same name as input, in downloads/ folder
        audio_path = Path(args.audio_file)
        output_path = Path('downloads') / audio_path.stem
    
    transcriber.save_transcription(result, output_path)
    
    print(f"\n[OK] Transcription saved!")


if __name__ == "__main__":
    main()
