"""
src/layout/reconstructor.py
===========================
Core document layout reconstruction engine.

The reconstructor takes the flat list of OCR results produced by
``OCRPipeline.run()`` and rebuilds the two-dimensional reading structure:

    1. **Box parsing** – convert raw dicts to typed ``TextBox`` objects and
       compute axis-aligned bounding rectangles.

    2. **Line grouping** – cluster boxes into horizontal text lines using a
       height-overlap heuristic.  Two boxes belong to the same line when their
       vertical intervals overlap by at least ``LayoutConfig.line_overlap_ratio``
       of the shorter box's height.

    3. **Column detection** – within each line, measure horizontal gaps between
       consecutive boxes.  Large gaps (≥ ``column_gap_ratio × median_width``)
       are treated as column separators.  Across the full page, boxes with
       similar x_min values are clustered into shared column anchors.

    4. **Table detection** – sequences of consecutive lines that share ≥
       ``table_col_count`` column anchors and span ≥ ``table_row_count`` rows
       are classified as tabular regions.

    5. **Plain-text rendering** – spacing between adjacent tokens on the same
       line is approximated by converting the pixel gap to a number of space
       characters proportional to the median character width.

    6. **Markdown table rendering** – detected table regions are rendered as
       GitHub-flavoured Markdown tables.

Usage
-----
    from src.layout.reconstructor import LayoutReconstructor
    from src.schemas.layout_config import LayoutConfig

    reconstructor = LayoutReconstructor()                 # default thresholds
    layout = reconstructor.reconstruct(pipeline_results)  # DocumentLayout

    print(layout.plain_text)
    print(layout.tables[0].markdown)

    import json
    print(json.dumps(layout.to_dict(), ensure_ascii=False, indent=2))
"""

from __future__ import annotations

import logging
import statistics
from typing import Dict, List, Optional, Set, Tuple

from src.layout.models import (
    BBox,
    DocumentLayout,
    LineGroup,
    TableRegion,
    TextBox,
)
from src.schemas.layout_config import LayoutConfig

logger = logging.getLogger(__name__)


# ── Public class ─────────────────────────────────────────────────────────────

class LayoutReconstructor:
    """
    Reconstructs document layout from a flat list of OCR results.

    Parameters
    ----------
    config : LayoutConfig | None
        Thresholds controlling line grouping, column detection, table
        identification, and plain-text spacing.  When *None*, sensible
        defaults are used (see ``LayoutConfig``).
    page_width : int
        Width of the source image in pixels (informational only).
    page_height : int
        Height of the source image in pixels (informational only).

    Example
    -------
    >>> from src.layout.reconstructor import LayoutReconstructor
    >>> layout = LayoutReconstructor().reconstruct(ocr_results)
    >>> print(layout.plain_text)
    """

    def __init__(
        self,
        config: Optional[LayoutConfig] = None,
        page_width: int = 0,
        page_height: int = 0,
    ) -> None:
        self.config = config or LayoutConfig()
        self.page_width = page_width
        self.page_height = page_height

    # ── Main entry-point ─────────────────────────────────────────────────────

    def reconstruct(
        self,
        ocr_results: List[Dict],
    ) -> DocumentLayout:
        """
        Full layout reconstruction pipeline.

        Parameters
        ----------
        ocr_results : list[dict]
            Output of ``OCRPipeline.run()``.  Each dict must contain at least
            ``"text"`` and ``"box"`` keys.

        Returns
        -------
        DocumentLayout
            Fully populated layout object with boxes, lines, tables, and
            rendered plain-text / Markdown.
        """
        if not ocr_results:
            logger.warning("reconstruct() received empty OCR results.")
            return DocumentLayout(page_width=self.page_width, page_height=self.page_height)

        # Step 1 – Parse raw OCR dicts into typed TextBox objects.
        boxes = self._parse_boxes(ocr_results)
        logger.debug("Parsed %d text boxes.", len(boxes))

        # Step 2 – Derive global metrics (median height/width) for thresholding.
        median_height, median_width = self._compute_metrics(boxes)
        logger.debug("Median box height=%.1f  width=%.1f", median_height, median_width)

        # Step 3 – Group boxes into horizontal text lines.
        lines = self._group_into_lines(boxes, median_height)
        logger.debug("Formed %d line groups.", len(lines))

        # Step 4 – Detect column anchors across the full page.
        column_anchors = self._detect_column_anchors(lines)
        logger.debug("Detected %d global column anchor(s).", len(column_anchors))

        # Step 5 – Assign column indices to each box; annotate line objects.
        self._assign_column_indices(lines, column_anchors)

        # Step 6 – Detect tabular regions.
        tables = self._detect_tables(lines, column_anchors)
        logger.debug("Detected %d table region(s).", len(tables))

        # Step 7 – Render table Markdown.
        for table in tables:
            table.markdown = self._render_table_markdown(table, lines)

        # Step 8 – Render plain text with spatial spacing.
        plain_text = self._render_plain_text(lines, median_width)

        # Step 9 – Build and return the DocumentLayout.
        layout = DocumentLayout(
            boxes=boxes,
            lines=lines,
            tables=tables,
            plain_text=plain_text,
            page_width=self.page_width,
            page_height=self.page_height,
        )
        return layout

    # ── Step 1: Parse boxes ───────────────────────────────────────────────────

    @staticmethod
    def _parse_boxes(ocr_results: List[Dict]) -> List[TextBox]:
        """Convert raw OCR dicts to ``TextBox`` objects, filtering empties."""
        boxes: List[TextBox] = []
        for result in ocr_results:
            if not result.get("text", "").strip():
                continue  # skip blank detections
            try:
                tb = TextBox.from_ocr_result(result)
                boxes.append(tb)
            except Exception as exc:
                logger.warning("Skipping malformed OCR result %s: %s", result, exc)
        return boxes

    # ── Step 2: Global metrics ────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(boxes: List[TextBox]) -> Tuple[float, float]:
        """Return (median_height, median_width) of all boxes."""
        if not boxes:
            return 10.0, 10.0
        heights = [b.height for b in boxes if b.height > 0]
        widths  = [b.width  for b in boxes if b.width  > 0]
        median_height = statistics.median(heights) if heights else 10.0
        median_width  = statistics.median(widths)  if widths  else 10.0
        return float(median_height), float(median_width)

    # ── Step 3: Line grouping ─────────────────────────────────────────────────

    def _group_into_lines(
        self,
        boxes: List[TextBox],
        median_height: float,
    ) -> List[LineGroup]:
        """
        Cluster boxes into horizontal text lines.

        Algorithm
        ---------
        Boxes are processed in top-to-bottom order.  For each box, we look for
        an existing line whose vertical interval overlaps with the box by at
        least ``line_overlap_ratio × min(box_height, line_height)``.  If
        found, the box is appended to that line.  Otherwise a new line is
        created.

        This is a greedy single-pass merge; it handles slight rotations and
        baselines that are not perfectly horizontal.
        """
        cfg = self.config
        gap_px = cfg.line_gap_tolerance * median_height

        # Sort top-to-bottom by y_min then left-to-right by x_min.
        sorted_boxes = sorted(boxes, key=lambda b: (b.y_min, b.x_min))

        lines: List[LineGroup] = []

        for box in sorted_boxes:
            matched_line: Optional[LineGroup] = None
            best_overlap = -1.0

            for line in lines:
                overlap = _vertical_overlap(box, line)
                min_dim = min(box.height, line.height) if line.height > 0 else box.height
                # Also accept boxes with a small absolute gap (gap_px tolerance).
                gap = max(box.y_min, line.y_min) - min(box.y_max, line.y_max)
                gap_ok = gap <= gap_px

                if (overlap >= cfg.line_overlap_ratio * min_dim or gap_ok) and overlap > best_overlap:
                    best_overlap = overlap
                    matched_line = line

            if matched_line is not None:
                matched_line.boxes.append(box)
                # Expand the line's vertical extent to cover the new box.
                matched_line.y_min = min(matched_line.y_min, box.y_min)
                matched_line.y_max = max(matched_line.y_max, box.y_max)
                matched_line.x_min = min(matched_line.x_min, box.x_min)
                matched_line.x_max = max(matched_line.x_max, box.x_max)
            else:
                # Start a new line group.
                line = LineGroup(
                    index=len(lines),
                    boxes=[box],
                    y_min=box.y_min,
                    y_max=box.y_max,
                    x_min=box.x_min,
                    x_max=box.x_max,
                )
                lines.append(line)

        # Sort each line's boxes left-to-right.
        for line in lines:
            line.boxes.sort(key=lambda b: b.x_min)
            # Update line_index back-reference on every box.
            for box in line.boxes:
                box.line_index = line.index

        # Final sort of lines top-to-bottom (by average y_min of their boxes).
        lines.sort(key=lambda ln: ln.y_min)
        # Re-index after sort.
        for i, line in enumerate(lines):
            line.index = i
            for box in line.boxes:
                box.line_index = i

        return lines

    # ── Step 4: Column anchor detection ──────────────────────────────────────

    def _detect_column_anchors(self, lines: List[LineGroup]) -> List[int]:
        """
        Find shared left-edge x-positions used as column separators.

        Strategy
        --------
        Collect the x_min of every text box.  Cluster nearby x values
        (within ``column_x_tolerance`` pixels) using a simple sweep.  Any
        cluster appearing in ≥ ``column_min_alignment`` lines is kept as a
        column anchor.

        Returns
        -------
        list[int]
            Sorted list of x-pixel positions representing column left edges.
        """
        cfg = self.config
        tol = cfg.column_x_tolerance

        # Gather (x_min, line_index) pairs.
        x_vals: List[Tuple[int, int]] = []
        for line in lines:
            for box in line.boxes:
                x_vals.append((box.x_min, line.index))

        if not x_vals:
            return []

        x_vals.sort(key=lambda t: t[0])

        # Sweep-merge nearby x values into clusters.
        clusters: List[List[Tuple[int, int]]] = []
        current: List[Tuple[int, int]] = [x_vals[0]]

        for xv, li in x_vals[1:]:
            if xv - current[-1][0] <= tol:
                current.append((xv, li))
            else:
                clusters.append(current)
                current = [(xv, li)]
        clusters.append(current)

        # Keep clusters that appear in enough distinct lines.
        anchors: List[int] = []
        for cluster in clusters:
            distinct_lines = len({li for _, li in cluster})
            if distinct_lines >= cfg.column_min_alignment:
                # Representative x = median of the cluster.
                median_x = int(statistics.median(xv for xv, _ in cluster))
                anchors.append(median_x)

        return sorted(anchors)

    # ── Step 5: Assign column indices ────────────────────────────────────────

    def _assign_column_indices(
        self,
        lines: List[LineGroup],
        column_anchors: List[int],
    ) -> None:
        """
        Tag each ``TextBox.column_index`` based on the nearest column anchor.

        Also populates ``LineGroup.column_anchors`` with the anchors that
        appear in each individual line.
        """
        if not column_anchors:
            return

        tol = self.config.column_x_tolerance * 2  # slightly relaxed for assignment

        for line in lines:
            found_anchors: Set[int] = set()
            for box in line.boxes:
                # Find the closest anchor within tolerance.
                best_idx: Optional[int] = None
                best_dist = float("inf")
                for col_i, anchor_x in enumerate(column_anchors):
                    dist = abs(box.x_min - anchor_x)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = col_i
                if best_dist <= tol and best_idx is not None:
                    box.column_index = best_idx
                    found_anchors.add(column_anchors[best_idx])

            line.column_anchors = sorted(found_anchors)

    # ── Step 6: Table detection ───────────────────────────────────────────────

    def _detect_tables(
        self,
        lines: List[LineGroup],
        column_anchors: List[int],
    ) -> List[TableRegion]:
        """
        Identify contiguous blocks of lines that form tabular regions.

        A block of consecutive lines is a table candidate when:
          - Each line contains entries in ≥ ``table_col_count`` columns.
          - The block spans ≥ ``table_row_count`` rows.
          - ≥ ``table_col_consistency × num_rows`` rows satisfy the column
            count criterion.
        """
        cfg = self.config

        if not column_anchors or len(lines) < cfg.table_row_count:
            return []

        def _line_col_count(line: LineGroup) -> int:
            """Number of distinct column anchors present in this line."""
            return len(line.column_anchors)

        # Scan for contiguous runs of "table-like" lines.
        tables: List[TableRegion] = []
        in_table = False
        run_start = 0

        for i, line in enumerate(lines):
            is_table_like = _line_col_count(line) >= cfg.table_col_count

            if is_table_like and not in_table:
                in_table = True
                run_start = i
            elif not is_table_like and in_table:
                # End of a candidate block.
                tables.extend(
                    self._validate_table_block(lines, run_start, i - 1, column_anchors)
                )
                in_table = False

        # Don't forget a run that reaches the last line.
        if in_table:
            tables.extend(
                self._validate_table_block(lines, run_start, len(lines) - 1, column_anchors)
            )

        return tables

    def _validate_table_block(
        self,
        lines: List[LineGroup],
        start: int,
        end: int,
        column_anchors: List[int],
    ) -> List[TableRegion]:
        """
        Validate and finalise a candidate table block (lines[start..end]).

        Returns either a single-element list with the ``TableRegion`` or an
        empty list if the block does not meet the minimum row/consistency
        thresholds.
        """
        cfg = self.config
        num_rows = end - start + 1

        if num_rows < cfg.table_row_count:
            return []

        block_lines = lines[start : end + 1]

        # Identify which column anchors actually appear in this block.
        present_anchors: Set[int] = set()
        for line in block_lines:
            present_anchors.update(line.column_anchors)

        block_anchors = sorted(present_anchors)
        num_cols = len(block_anchors)

        if num_cols < cfg.table_col_count:
            return []

        # Check consistency: fraction of rows with ≥ table_col_count columns.
        multi_col_rows = sum(
            1 for ln in block_lines if len(ln.column_anchors) >= cfg.table_col_count
        )
        consistency = multi_col_rows / num_rows

        if consistency < cfg.table_col_consistency:
            return []

        # Mark lines as table rows.
        for line in block_lines:
            line.is_table_row = True

        region = TableRegion(
            line_indices=list(range(start, end + 1)),
            column_anchors=block_anchors,
            num_rows=num_rows,
            num_cols=num_cols,
        )
        return [region]

    # ── Step 7: Markdown rendering ────────────────────────────────────────────

    def _render_table_markdown(
        self,
        table: TableRegion,
        lines: List[LineGroup],
    ) -> str:
        """
        Render a ``TableRegion`` as a GitHub-Flavoured Markdown table string.

        Each detected column anchor becomes one Markdown column.  Boxes are
        placed in the column whose anchor is closest to the box's x_min.
        Empty cells are represented as a single space.
        """
        col_anchors = table.column_anchors
        num_cols = len(col_anchors)

        if num_cols == 0:
            return ""

        rows: List[List[str]] = []

        for line_idx in table.line_indices:
            line = lines[line_idx]
            # Build an empty row.
            row: List[str] = ["" for _ in range(num_cols)]

            for box in line.boxes:
                # Assign box to nearest column.
                col_i = _nearest_anchor_index(box.x_min, col_anchors)
                cell_text = row[col_i]
                row[col_i] = (cell_text + " " + box.text).strip() if cell_text else box.text

            rows.append(row)

        if not rows:
            return ""

        # Use the first row as header if it differs from subsequent rows.
        header = rows[0]
        data_rows = rows[1:]

        def _fmt_row(cells: List[str]) -> str:
            # Escape pipe characters inside cell text.
            escaped = [c.replace("|", "\\|") if c else " " for c in cells]
            return "| " + " | ".join(escaped) + " |"

        separator = "| " + " | ".join("---" for _ in range(num_cols)) + " |"

        md_lines = [_fmt_row(header), separator]
        md_lines += [_fmt_row(row) for row in data_rows]
        return "\n".join(md_lines)

    # ── Step 8: Plain-text rendering ──────────────────────────────────────────

    def _render_plain_text(
        self,
        lines: List[LineGroup],
        median_width: float,
    ) -> str:
        """
        Reconstruct a plain-text representation with spatial spacing.

        For each line:
          - Boxes are already sorted left-to-right.
          - The horizontal gap between consecutive boxes is converted to a
            number of space characters using ``space_width_ratio``.
          - Lines are joined with newlines.

        The result approximates the visual column alignment of the original
        document.
        """
        cfg = self.config

        # Estimate the median character width from box dimensions.
        char_widths: List[float] = []
        for line in lines:
            for box in line.boxes:
                n_chars = len(box.text)
                if n_chars > 0:
                    char_widths.append(box.width / n_chars)

        median_char_width = statistics.median(char_widths) if char_widths else max(median_width / 10, 1.0)
        median_char_width = max(median_char_width, 1.0)  # avoid division by zero

        rendered_lines: List[str] = []

        for line in lines:
            if not line.boxes:
                continue

            parts: List[str] = []
            prev_box: Optional[TextBox] = None

            for box in line.boxes:
                if prev_box is None:
                    parts.append(box.text)
                else:
                    gap_px = box.x_min - prev_box.x_max
                    n_spaces = int(gap_px / median_char_width * cfg.space_width_ratio)
                    n_spaces = max(cfg.min_spaces, min(n_spaces, cfg.max_spaces))
                    parts.append(" " * n_spaces + box.text)
                prev_box = box

            line.text = "".join(parts)
            rendered_lines.append(line.text)

        return "\n".join(rendered_lines)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _vertical_overlap(box: TextBox, line: LineGroup) -> float:
    """
    Compute the pixel height of vertical overlap between a box and a line.

    Returns a negative value (gap size) when they don't overlap.
    """
    overlap_top    = max(box.y_min, line.y_min)
    overlap_bottom = min(box.y_max, line.y_max)
    return float(overlap_bottom - overlap_top)


def _nearest_anchor_index(x: int, anchors: List[int]) -> int:
    """Return the index of the closest anchor to ``x``."""
    best_i = 0
    best_dist = abs(x - anchors[0])
    for i, ax in enumerate(anchors[1:], start=1):
        dist = abs(x - ax)
        if dist < best_dist:
            best_dist = dist
            best_i = i
    return best_i
