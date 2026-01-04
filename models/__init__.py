"""Model definitions for FAS system."""

from .backbones import get_backbone
from .fusion import FeatureFusion

__all__ = ["get_backbone", "FeatureFusion"]
