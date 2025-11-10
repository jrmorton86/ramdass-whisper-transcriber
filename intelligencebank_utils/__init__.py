"""Intelligence Bank utilities for asset management."""

# Add parent directory to path for imports
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings

__all__ = ['settings']
