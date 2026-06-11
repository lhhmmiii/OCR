"""
src/layout
==========
Document layout reconstruction from OCR bounding boxes.

Public surface
--------------
    LayoutReconstructor   – main class
    LayoutConfig          – configurable thresholds
    DocumentLayout        – structured output (JSON-serialisable)
"""

from src.layout.reconstructor import LayoutReconstructor
from src.layout.models import (
    BBox,
    TextBox,
    LineGroup,
    TableRegion,
    DocumentLayout,
)
from src.schemas.layout_config import LayoutConfig

__all__ = [
    "LayoutReconstructor",
    "LayoutConfig",
    "BBox",
    "TextBox",
    "LineGroup",
    "TableRegion",
    "DocumentLayout",
]
