"""
Montage AI - AI-Powered Video Montage Creation

Usage:
    from montage_ai import create_montage
    create_montage(variant_id=1)
"""

__version__ = "0.3.0"

from .editor import create_montage

__all__ = ["create_montage"]
