"""
Pipeline Package - Audio Transcription Pipeline

This package contains the complete audio transcription pipeline including:
- Whisper transcription with vocabulary enhancement
- Claude refinement
- Post-processing (embeddings, Comprehend, DB, S3)
- Batch processing utilities
- Worker pool for GPU-accelerated processing
"""

import sys
from pathlib import Path

# Add this package to the Python path for relative imports
PACKAGE_DIR = Path(__file__).parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

# Also add parent for cross-package imports
PACKAGES_DIR = PACKAGE_DIR.parent
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))
