"""
src/layout/models.py
====================
Typed data-model classes for document layout reconstruction output.

All classes are plain Python dataclasses so they can be trivially converted to
JSON via ``dataclasses.asdict()``.  No third-party dependencies are required.

Hierarchy
---------
    DocumentLayout
    ├── List[TextBox]       (every detected box, flat – mirrors OCR output)
    └── List[LineGroup]     (boxes clustered into text lines)
        └── each LineGroup owns its TextBox list

Tabular regions are identified as a separate list of ``TableRegion`` objects
that cross-reference existing ``LineGroup`` indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Primitive types
# ---------------------------------------------------------------------------

# Four corner points returned by PaddleOCR: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
# Stored as a list-of-lists for JSON compatibility.
BBox = List[List[int]]


# ---------------------------------------------------------------------------
# TextBox – a single OCR result with full spatial metadata
# ---------------------------------------------------------------------------

@dataclass
class TextBox:
    """
    A single recognised text region.

    Attributes
    ----------
    text : str
        Recognised text string.
    bbox : BBox
        Four corner points [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] from PaddleOCR.
    x_min, x_max : int
        Horizontal extent of the bounding box (axis-aligned).
    y_min, y_max : int
        Vertical extent of the bounding box (axis-aligned).
    width, height : int
        Derived dimensions.
    detection_score : float | None
        Confidence score from PaddleOCR detection head.
    recognition_score : float | None
        Confidence score from VietOCR recognition head.
    line_index : int
        Index of the ``LineGroup`` this box belongs to (set by reconstructor).
    column_index : int | None
        Detected column index within its line (set by reconstructor).
    """

    text: str
    bbox: BBox

    # Axis-aligned bounds (derived from bbox corners)
    x_min: int = 0
    x_max: int = 0
    y_min: int = 0
    y_max: int = 0
    width: int = 0
    height: int = 0

    # OCR confidence
    detection_score: Optional[float] = None
    recognition_score: Optional[float] = None

    # Layout metadata (filled in by LayoutReconstructor)
    line_index: int = -1
    column_index: Optional[int] = None

    @classmethod
    def from_ocr_result(cls, result: dict) -> "TextBox":
        """
        Construct a TextBox from a single ``OCRPipeline.run()`` output dict.

        Expected dict keys: ``text``, ``box``, ``detection_score``,
        ``recognition_score``.
        """
        box: BBox = result["box"]

        # Compute axis-aligned bounding rectangle from the 4 corner points.
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        return cls(
            text=result.get("text", ""),
            bbox=box,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            width=x_max - x_min,
            height=y_max - y_min,
            detection_score=result.get("detection_score"),
            recognition_score=result.get("recognition_score"),
        )

    @property
    def x_center(self) -> float:
        """Horizontal centre of the axis-aligned box."""
        return (self.x_min + self.x_max) / 2.0

    @property
    def y_center(self) -> float:
        """Vertical centre of the axis-aligned box."""
        return (self.y_min + self.y_max) / 2.0


# ---------------------------------------------------------------------------
# LineGroup – a cluster of TextBoxes that form one logical text line
# ---------------------------------------------------------------------------

@dataclass
class LineGroup:
    """
    A group of TextBoxes that lie on the same horizontal text line.

    Attributes
    ----------
    index : int
        Zero-based line index (reading order, top-to-bottom).
    boxes : list[TextBox]
        Constituent text boxes sorted left-to-right.
    y_min, y_max : int
        Vertical extent of the line (union of all box extents).
    x_min, x_max : int
        Horizontal extent of the line.
    text : str
        Reconstructed line text with spacing preserved.
    is_table_row : bool
        True when this line has been classified as part of a table.
    column_anchors : list[int]
        X positions of detected column anchors within this line.
    """

    index: int
    boxes: List[TextBox] = field(default_factory=list)

    # Spatial extent (computed after all boxes are added)
    y_min: int = 0
    y_max: int = 0
    x_min: int = 0
    x_max: int = 0

    # Reconstructed text for this line
    text: str = ""

    # Table classification
    is_table_row: bool = False

    # Detected column anchor x-positions
    column_anchors: List[int] = field(default_factory=list)

    def update_bounds(self) -> None:
        """Recompute spatial extent from current ``boxes`` list."""
        if not self.boxes:
            return
        self.y_min = min(b.y_min for b in self.boxes)
        self.y_max = max(b.y_max for b in self.boxes)
        self.x_min = min(b.x_min for b in self.boxes)
        self.x_max = max(b.x_max for b in self.boxes)

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def width(self) -> int:
        return self.x_max - self.x_min


# ---------------------------------------------------------------------------
# TableRegion – a contiguous block of lines identified as tabular content
# ---------------------------------------------------------------------------

@dataclass
class TableRegion:
    """
    A contiguous block of ``LineGroup`` objects inferred to be a table.

    Attributes
    ----------
    line_indices : list[int]
        Indices into ``DocumentLayout.lines`` that make up this table.
    column_anchors : list[int]
        Shared x-positions used as column separators.
    num_rows : int
        Number of rows (= len(line_indices)).
    num_cols : int
        Number of detected columns.
    markdown : str
        Rendered Markdown table string.
    """

    line_indices: List[int] = field(default_factory=list)
    column_anchors: List[int] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    markdown: str = ""


# ---------------------------------------------------------------------------
# DocumentLayout – top-level output of the reconstructor
# ---------------------------------------------------------------------------

@dataclass
class DocumentLayout:
    """
    Complete layout-aware representation of a document page.

    Attributes
    ----------
    boxes : list[TextBox]
        All detected text boxes (flat list, mirrors raw OCR output order).
    lines : list[LineGroup]
        Boxes grouped into horizontal text lines, sorted top-to-bottom.
    tables : list[TableRegion]
        Identified tabular regions with rendered Markdown.
    plain_text : str
        Full document text with spacing and line breaks preserved.
    page_width, page_height : int
        Image dimensions used during reconstruction (0 if unknown).
    """

    boxes: List[TextBox] = field(default_factory=list)
    lines: List[LineGroup] = field(default_factory=list)
    tables: List[TableRegion] = field(default_factory=list)

    plain_text: str = ""

    page_width: int = 0
    page_height: int = 0

    def to_dict(self) -> dict:
        """
        Return a JSON-serialisable dict representation.

        The nested dataclasses are converted recursively.  ``BBox`` values
        remain as lists-of-lists (already JSON-compatible).
        """
        import dataclasses

        def _convert(obj):
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            return obj

        return _convert(self)
