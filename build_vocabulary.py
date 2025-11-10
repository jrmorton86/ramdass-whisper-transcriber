#!/usr/bin/env python3
"""
Build custom vocabulary for Whisper transcription from AWS Transcribe exports
and keyword analysis CSVs.

This script:
1. Parses AWS Transcribe custom vocabulary (input.txt)
2. Loads filter words to exclude (filter.txt)
3. Incorporates top words/phrases from CSV analysis
4. Generates Whisper-compatible vocabulary files
"""

import csv
import json
from pathlib import Path
from collections import defaultdict


class VocabularyBuilder:
    def __init__(self, keyword_dir="keyword_lists"):
        self.keyword_dir = Path(keyword_dir)
        self.custom_vocab = {}  # phrase -> {sounds_like, ipa, display_as}
        self.filter_words = set()
        self.top_words = []
        self.top_phrases = []
        
    def load_aws_vocabulary(self, filename="input.txt"):
        """Load AWS Transcribe custom vocabulary format"""
        filepath = self.keyword_dir / filename
        
        if not filepath.exists():
            print(f"⚠️  Vocabulary file not found: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip header
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) < 4:
                continue
            
            phrase = parts[0].strip()
            sounds_like = parts[1].strip()
            ipa = parts[2].strip()
            display_as = parts[3].strip()
            
            self.custom_vocab[phrase.lower()] = {
                'phrase': phrase,
                'sounds_like': sounds_like,
                'ipa': ipa,
                'display_as': display_as
            }
        
        print(f"✓ Loaded {len(self.custom_vocab)} custom vocabulary entries")
    
    def load_filter_words(self, filename="filter.txt"):
        """Load words to filter/ignore"""
        filepath = self.keyword_dir / filename
        
        if not filepath.exists():
            print(f"⚠️  Filter file not found: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            self.filter_words = set(line.strip().lower() for line in f if line.strip())
        
        print(f"✓ Loaded {len(self.filter_words)} filter words")
    
    def load_top_words_csv(self, pattern="embedding_top_words_*.csv", top_n=None):
        """Load top words from CSV analysis (all words if top_n is None)"""
        files = list(self.keyword_dir.glob(pattern))
        
        if not files:
            print(f"⚠️  No word CSV files found matching: {pattern}")
            return
        
        # Use most recent file
        filepath = sorted(files)[-1]
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if top_n is not None and i >= top_n:
                    break
                word = row['Word'].strip()
                if word.lower() not in self.filter_words:
                    self.top_words.append({
                        'word': word,
                        'count': int(row['Count']),
                        'percentage': row['Percentage']
                    })
        
        print(f"✓ Loaded {len(self.top_words)} top words from {filepath.name}")
    
    def load_top_phrases_csv(self, pattern="embedding_top_phrases_*.csv", top_n=None):
        """Load top phrases from CSV analysis (all phrases if top_n is None)"""
        files = list(self.keyword_dir.glob(pattern))
        
        if not files:
            print(f"⚠️  No phrase CSV files found matching: {pattern}")
            return
        
        # Use most recent file
        filepath = sorted(files)[-1]
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if top_n is not None and i >= top_n:
                    break
                phrase = row['Phrase'].strip()
                # Skip if phrase is in filter words
                if not any(word in self.filter_words for word in phrase.lower().split()):
                    self.top_phrases.append({
                        'phrase': phrase,
                        'count': int(row['Count']),
                        'percentage': row['Percentage']
                    })
        
        print(f"✓ Loaded {len(self.top_phrases)} top phrases from {filepath.name}")
    
    def generate_whisper_vocab(self, output_file="whisper_vocabulary.json"):
        """Generate Whisper-compatible vocabulary JSON"""
        output_path = self.keyword_dir / output_file
        
        vocab_data = {
            'custom_vocabulary': [],
            'filter_words': list(self.filter_words),
            'top_words': self.top_words,  # All words
            'top_phrases': self.top_phrases,  # All phrases
            'metadata': {
                'total_custom_entries': len(self.custom_vocab),
                'total_filter_words': len(self.filter_words),
                'total_top_words': len(self.top_words),
                'total_top_phrases': len(self.top_phrases),
                'description': 'Custom vocabulary for Ram Dass transcriptions'
            }
        }
        
        # Add custom vocabulary with pronunciation hints
        for key, entry in self.custom_vocab.items():
            vocab_entry = {
                'phrase': entry['phrase'],
                'display_as': entry['display_as']
            }
            
            if entry['sounds_like']:
                vocab_entry['sounds_like'] = entry['sounds_like']
            if entry['ipa']:
                vocab_entry['ipa'] = entry['ipa']
            
            vocab_data['custom_vocabulary'].append(vocab_entry)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Generated Whisper vocabulary: {output_path}")
        return output_path
    
    def generate_replacement_map(self, output_file="replacement_map.json"):
        """Generate comprehensive phrase replacement map for post-processing"""
        output_path = self.keyword_dir / output_file
        
        replacement_map = {}
        
        for key, entry in self.custom_vocab.items():
            phrase_lower = entry['phrase'].lower()
            display = entry['display_as']
            sounds_like = entry['sounds_like'].lower() if entry['sounds_like'] else None
            
            # 1. Add the phrase itself (all case variations)
            replacement_map[phrase_lower] = display
            replacement_map[phrase_lower.title()] = display
            replacement_map[phrase_lower.upper()] = display
            
            # 2. Add variations without hyphens/spaces
            if '-' in phrase_lower:
                replacement_map[phrase_lower.replace('-', ' ')] = display
                replacement_map[phrase_lower.replace('-', '')] = display
            
            if ' ' in phrase_lower:
                replacement_map[phrase_lower.replace(' ', '')] = display
                replacement_map[phrase_lower.replace(' ', '-')] = display
            
            # 3. Add phonetic/sounds-like variations
            if sounds_like:
                replacement_map[sounds_like] = display
                replacement_map[sounds_like.replace('-', ' ')] = display
                replacement_map[sounds_like.replace('-', '')] = display
                replacement_map[sounds_like.title()] = display
            
            # 4. Add common misspellings/mishearings for specific terms
            # These are patterns Whisper commonly gets wrong
            common_errors = {
                'ram dass': ['rom das', 'romdas', 'ram das', 'ramdas', 'rahm das', 'ron das'],
                'maharaj-ji': ['maharaja', 'maharaji', 'maharajee', 'mahraj', 'mahrajji'],
                'neem karoli baba': ['neem karoli', 'nim karoli', 'neem coroli', 'neemkaroli'],
                'hanuman': ['hanaman', 'hannuman', 'hunuman'],
                'vrindavan': ['vrindaban', 'brindavan', 'vrindabhan'],
                'rishikesh': ['rishikash', 'rishkesh', 'rishikeshe'],
                'bhakti': ['bakti', 'bhakty', 'bacti'],
                'kirtan': ['keetan', 'keertan', 'kirten'],
                'satsang': ['satsong', 'sat sang', 'satseng'],
                'pranayama': ['pranyama', 'pranayam', 'pranayuma'],
                'kundalini': ['kundalani', 'kundolini', 'kundelini'],
                'darshan': ['darshun', 'darshon', 'darshaan'],
                'prasad': ['prasaad', 'prasod', 'prashad'],
                'vipassana': ['vipassana', 'vipassna', 'vipasana'],
                'bodhisattva': ['bodisattva', 'bodhisatva', 'boddhisattva'],
                'kabbalah': ['kabala', 'cabalah', 'kabbala', 'cabala'],
                'shekhinah': ['shekinah', 'shechina', 'shechinah'],
            }
            
            # 5. Add standalone common errors (not tied to input.txt)
            standalone_errors = {
                'ouspensky': ['lispensky', 'ouspenski', 'uspensky', 'uspenski'],
                'timothy leary': ['timothy leery', 'timothy o\'leary'],
                'gurdjieff': ['gurdjief', 'gurdgief', 'gurdjeff'],
            }
            
            for correct, variations in standalone_errors.items():
                for variant in variations:
                    replacement_map[variant] = correct.title()
                    replacement_map[variant.title()] = correct.title()
                    replacement_map[variant.upper()] = correct.upper()
            
            for correct, variations in common_errors.items():
                if correct in phrase_lower:
                    for variant in variations:
                        replacement_map[variant] = display
                        replacement_map[variant.title()] = display
        
        # Sort by length (longest first) for better replacement order
        sorted_map = dict(sorted(replacement_map.items(), key=lambda x: len(x[0]), reverse=True))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_map, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Generated replacement map: {output_path} ({len(sorted_map)} rules)")
        return output_path
    
    def generate_summary_report(self, output_file="vocabulary_summary.txt"):
        """Generate human-readable summary"""
        output_path = self.keyword_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("CUSTOM VOCABULARY SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Custom Vocabulary Entries: {len(self.custom_vocab)}\n")
            f.write(f"Filter Words: {len(self.filter_words)}\n")
            f.write(f"Top Words Analyzed: {len(self.top_words)}\n")
            f.write(f"Top Phrases Analyzed: {len(self.top_phrases)}\n\n")
            
            f.write("-"*70 + "\n")
            f.write("CUSTOM VOCABULARY (with pronunciation)\n")
            f.write("-"*70 + "\n")
            for entry in sorted(self.custom_vocab.values(), key=lambda x: x['phrase']):
                f.write(f"\n{entry['phrase']}")
                if entry['sounds_like']:
                    f.write(f" ({entry['sounds_like']})")
                f.write(f" → {entry['display_as']}\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("FILTER WORDS (to ignore)\n")
            f.write("-"*70 + "\n")
            f.write(", ".join(sorted(self.filter_words)) + "\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("TOP 20 MOST COMMON WORDS\n")
            f.write("-"*70 + "\n")
            for i, item in enumerate(self.top_words[:20], 1):
                f.write(f"{i:2d}. {item['word']:20s} ({item['count']:4d} times, {item['percentage']})\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("TOP 20 MOST COMMON PHRASES\n")
            f.write("-"*70 + "\n")
            for i, item in enumerate(self.top_phrases[:20], 1):
                f.write(f"{i:2d}. {item['phrase']:30s} ({item['count']:4d} times, {item['percentage']})\n")
        
        print(f"✓ Generated summary report: {output_path}")
        return output_path
    
    def build_all(self):
        """Build all vocabulary files"""
        print("\n" + "="*70)
        print("BUILDING CUSTOM VOCABULARY")
        print("="*70 + "\n")
        
        self.load_aws_vocabulary()
        self.load_filter_words()
        self.load_top_words_csv()
        self.load_top_phrases_csv()
        
        print("\n" + "-"*70)
        print("GENERATING OUTPUT FILES")
        print("-"*70 + "\n")
        
        self.generate_whisper_vocab()
        self.generate_replacement_map()
        self.generate_summary_report()
        
        print("\n" + "="*70)
        print("✓ VOCABULARY BUILD COMPLETE")
        print("="*70 + "\n")


def main():
    builder = VocabularyBuilder()
    builder.build_all()
    
    print("\nGenerated files:")
    print("  • whisper_vocabulary.json - Full vocabulary for Whisper")
    print("  • replacement_map.json - Post-processing replacements")
    print("  • vocabulary_summary.txt - Human-readable summary")
    print("\nUse these files in your transcription scripts for better accuracy!")


if __name__ == "__main__":
    main()
