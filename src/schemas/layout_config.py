"""
src/schemas/layout_config.py
============================
Configurable thresholds for document layout reconstruction.

All pixel/ratio values are intentionally kept separate from OCRConfig so the
layout engine can be tuned independently of detection/recognition quality.
"""

from dataclasses import dataclass, field


@dataclass
class LayoutConfig:
    """
    Thresholds that control how the layout reconstructor groups and formats
    detected text boxes.

    Line grouping
    -------------
    line_overlap_ratio : float
        Two boxes are merged into the same line when the vertical overlap
        between them (relative to the shorter box's height) is ≥ this value.
        Higher values → stricter line grouping (fewer boxes per line).
        Range: 0.0 – 1.0.  Default: 0.5

    line_gap_tolerance : float
        Extra vertical gap (as a fraction of the median box height) still
        allowed when ``line_overlap_ratio`` alone is too strict.  Useful for
        slightly skewed or condensed documents.
        Range: 0.0 – 1.0.  Default: 0.3

    Column detection
    ----------------
    column_gap_ratio : float
        Minimum horizontal gap between two adjacent boxes (relative to the
        median box width) required to infer a column break.
        A column break causes extra spaces to be inserted in plain-text output.
        Range: 0.0 – 5.0.  Default: 1.5

    column_min_alignment : int
        Minimum number of boxes that must share a similar x-left coordinate
        before that x position is considered a column anchor.
        Higher values → only very consistent columns are detected.
        Default: 2

    column_x_tolerance : int
        Maximum pixel deviation allowed when deciding whether two boxes share
        the same left-edge (column anchor).
        Default: 15  (pixels)

    Table detection
    ---------------
    table_col_count : int
        Minimum number of distinct column anchors required for a block of lines
        to be classified as a table/tabular region.
        Default: 2

    table_row_count : int
        Minimum number of rows a block must have before it can be labelled a
        table.
        Default: 2

    table_col_consistency : float
        Fraction of rows in a candidate table block that must contain an entry
        in at least ``table_col_count`` columns.
        Range: 0.0 – 1.0.  Default: 0.6

    Plain-text rendering
    --------------------
    space_width_ratio : float
        Number of space characters inserted per unit of ``median_char_width``
        of gap between two boxes on the same line.
        Tune this to control how dense/sparse the reconstructed plain text is.
        Default: 0.5

    min_spaces : int
        Minimum spaces between two tokens on the same line (even when the
        measured gap is small).
        Default: 1

    max_spaces : int
        Cap on spaces inserted between tokens to avoid runaway whitespace.
        Default: 20
    """

    # ── Line grouping ────────────────────────────────────────────────────────
    line_overlap_ratio: float = 0.5
    line_gap_tolerance: float = 0.3

    # ── Column detection ─────────────────────────────────────────────────────
    column_gap_ratio: float = 1.5
    column_min_alignment: int = 2
    column_x_tolerance: int = 15

    # ── Table detection ──────────────────────────────────────────────────────
    table_col_count: int = 2
    table_row_count: int = 2
    table_col_consistency: float = 0.6

    # ── Plain-text spacing ───────────────────────────────────────────────────
    space_width_ratio: float = 0.5
    min_spaces: int = 1
    max_spaces: int = 20
