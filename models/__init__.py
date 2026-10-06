"""
Models package initialization
"""
from pathlib import Path

# Create pretrained directory
PRETRAINED_DIR = Path(__file__).parent / 'pretrained'
PRETRAINED_DIR.mkdir(exist_ok=True)

__all__ = ['disease_detector', 'data_processor']
