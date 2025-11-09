#!/usr/bin/env python3
"""
Speaker configuration and identification system.
Allows mapping of detected speakers to known identities.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import json
from pathlib import Path


@dataclass
class Speaker:
    """Represents a known speaker"""
    id: str
    name: str
    aliases: List[str] = None
    voice_characteristics: Dict = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.voice_characteristics is None:
            self.voice_characteristics = {}


class SpeakerConfig:
    """Manages speaker configuration and identification"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.speakers: Dict[str, Speaker] = {}
        self.diarization_mapping: Dict[str, str] = {}  # Maps SPEAKER_XX to speaker name
        
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
    
    def add_speaker(self, speaker_id: str, name: str, aliases: List[str] = None):
        """Add a new speaker to the config"""
        self.speakers[speaker_id] = Speaker(
            id=speaker_id,
            name=name,
            aliases=aliases or []
        )
    
    def set_diarization_mapping(self, diarization_speaker: str, config_speaker: str):
        """Map a diarization speaker (SPEAKER_00) to a known speaker"""
        if config_speaker not in self.speakers:
            print(f"Warning: Speaker '{config_speaker}' not found in config")
            return
        
        self.diarization_mapping[diarization_speaker] = config_speaker
    
    def get_speaker_name(self, diarization_speaker: str) -> str:
        """Get the real name for a diarization speaker. Falls back to SPEAKER_XX if no mapping."""
        config_speaker = self.diarization_mapping.get(diarization_speaker)
        if config_speaker and config_speaker in self.speakers:
            return self.speakers[config_speaker].name
        # Return the original diarization label if no mapping exists
        return diarization_speaker
    
    def save_config(self, filepath: str):
        """Save speaker configuration to JSON"""
        config_data = {
            "speakers": {
                sid: {
                    "name": s.name,
                    "aliases": s.aliases,
                    "voice_characteristics": s.voice_characteristics
                }
                for sid, s in self.speakers.items()
            },
            "diarization_mapping": self.diarization_mapping
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✓ Speaker config saved to {filepath}")
    
    def load_config(self, filepath: str):
        """Load speaker configuration from JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        self.speakers = {
            sid: Speaker(
                id=sid,
                name=data["name"],
                aliases=data.get("aliases", []),
                voice_characteristics=data.get("voice_characteristics", {})
            )
            for sid, data in config_data.get("speakers", {}).items()
        }
        
        self.diarization_mapping = config_data.get("diarization_mapping", {})
        print(f"✓ Speaker config loaded from {filepath}")
    
    def auto_map_speakers(self, detected_speakers: List[str], known_names: List[str]):
        """
        Automatically map detected speakers to known names in order.
        Maps first detected speaker to first name, etc.
        
        Args:
            detected_speakers: List of diarization labels (e.g., ['SPEAKER_00', 'SPEAKER_01'])
            known_names: List of speaker names (e.g., ['Ram Dass', 'Host', 'Caller'])
        """
        detected_sorted = sorted(detected_speakers)
        
        for i, diarization_speaker in enumerate(detected_sorted):
            if i < len(known_names):
                # Use provided name
                speaker_name = known_names[i]
                speaker_id = speaker_name.lower().replace(' ', '_').replace('-', '_')
                
                # Add speaker if not already in config
                if speaker_id not in self.speakers:
                    self.add_speaker(speaker_id, speaker_name)
                
                # Create mapping
                self.set_diarization_mapping(diarization_speaker, speaker_id)
            else:
                # More speakers detected than names provided
                # Create generic names like "Speaker 4", "Speaker 5"
                speaker_name = f"Speaker {i + 1}"
                speaker_id = f"speaker_{i + 1}"
                
                if speaker_id not in self.speakers:
                    self.add_speaker(speaker_id, speaker_name)
                
                self.set_diarization_mapping(diarization_speaker, speaker_id)
    
    def list_speakers(self):
        """List all configured speakers"""
        if not self.speakers:
            print("No speakers configured. Will use default SPEAKER_XX labels.")
            return
        
        print("\nConfigured Speakers:")
        print("-"*60)
        for sid, speaker in self.speakers.items():
            print(f"  {speaker.name:20} (id: {sid})")
            if speaker.aliases:
                print(f"    Aliases: {', '.join(speaker.aliases)}")
        
        if self.diarization_mapping:
            print("\nDiarization Mappings:")
            print("-"*60)
            for dia, config in self.diarization_mapping.items():
                speaker = self.speakers.get(config)
                if speaker:
                    print(f"  {dia:15} -> {speaker.name}")
        print()
