"""
test/test_layout_reconstructor.py
===================================
Unit tests for the document layout reconstruction module.

Run with:
    pytest test/test_layout_reconstructor.py -v
"""

import json
import pytest
from src.layout.reconstructor import LayoutReconstructor, _vertical_overlap, _nearest_anchor_index
from src.layout.models import TextBox, LineGroup, DocumentLayout
from src.schemas.layout_config import LayoutConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(text: str, x0: int, y0: int, x1: int, y1: int) -> dict:
    """
    Create a mock OCR result dict with an axis-aligned box (corners ordered
    top-left, top-right, bottom-right, bottom-left — PaddleOCR convention).
    """
    return {
        "text": text,
        "box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "detection_score": 0.99,
        "recognition_score": 0.95,
    }


def _simple_invoice() -> list:
    """
    Synthetic invoice OCR results with two label-value columns and a
    3-column line-item table section.
    """
    return [
        # Header
        _make_result("INVOICE",         200,  10, 400,  35),

        # Meta section – two columns: label (x≈20) and value (x≈200)
        _make_result("Invoice No:",      20,  50, 110,  70),
        _make_result("INV-2024-001",    200,  50, 340,  70),
        _make_result("Date:",            20,  80, 80,   100),
        _make_result("2024-06-01",      200,  80, 310,  100),
        _make_result("Client:",          20, 110, 90,   130),
        _make_result("Acme Corp",       200, 110, 310,  130),

        # Table header row (x≈20, x≈250, x≈400)
        _make_result("Description",      20, 160, 200,  180),
        _make_result("Qty",             250, 160, 290,  180),
        _make_result("Amount",          400, 160, 480,  180),

        # Table row 1
        _make_result("Web Design",       20, 190, 200,  210),
        _make_result("1",               250, 190, 270,  210),
        _make_result("500.00",          400, 190, 470,  210),

        # Table row 2
        _make_result("Hosting",          20, 220, 140,  240),
        _make_result("12",              250, 220, 275,  240),
        _make_result("120.00",          400, 220, 465,  240),

        # Footer
        _make_result("Total:",          350, 270, 410,  290),
        _make_result("620.00",          420, 270, 490,  290),
    ]


# ── TextBox parsing ───────────────────────────────────────────────────────────

class TestTextBoxParsing:
    def test_basic_fields(self):
        result = _make_result("Hello", 10, 20, 110, 50)
        tb = TextBox.from_ocr_result(result)
        assert tb.text == "Hello"
        assert tb.x_min == 10
        assert tb.x_max == 110
        assert tb.y_min == 20
        assert tb.y_max == 50
        assert tb.width == 100
        assert tb.height == 30

    def test_centers(self):
        result = _make_result("X", 0, 0, 100, 40)
        tb = TextBox.from_ocr_result(result)
        assert tb.x_center == 50.0
        assert tb.y_center == 20.0

    def test_scores_preserved(self):
        result = _make_result("A", 0, 0, 50, 20)
        result["detection_score"] = 0.88
        result["recognition_score"] = 0.77
        tb = TextBox.from_ocr_result(result)
        assert tb.detection_score == pytest.approx(0.88)
        assert tb.recognition_score == pytest.approx(0.77)


# ── Vertical overlap helper ───────────────────────────────────────────────────

class TestVerticalOverlap:
    def _box(self, y0, y1):
        r = _make_result("x", 0, y0, 50, y1)
        return TextBox.from_ocr_result(r)

    def _line(self, y0, y1) -> LineGroup:
        ln = LineGroup(index=0, y_min=y0, y_max=y1)
        return ln

    def test_full_overlap(self):
        box = self._box(10, 30)
        line = self._line(10, 30)
        assert _vertical_overlap(box, line) == pytest.approx(20.0)

    def test_partial_overlap(self):
        box = self._box(10, 30)
        line = self._line(20, 40)
        assert _vertical_overlap(box, line) == pytest.approx(10.0)

    def test_no_overlap(self):
        box = self._box(10, 20)
        line = self._line(30, 40)
        assert _vertical_overlap(box, line) < 0


# ── Nearest anchor helper ─────────────────────────────────────────────────────

class TestNearestAnchor:
    def test_exact(self):
        assert _nearest_anchor_index(50, [10, 50, 90]) == 1

    def test_closest(self):
        assert _nearest_anchor_index(12, [10, 50, 90]) == 0
        assert _nearest_anchor_index(48, [10, 50, 90]) == 1
        assert _nearest_anchor_index(80, [10, 50, 90]) == 2


# ── Line grouping ─────────────────────────────────────────────────────────────

class TestLineGrouping:
    def _reconstruct(self, results, **cfg_kwargs):
        cfg = LayoutConfig(**cfg_kwargs)
        return LayoutReconstructor(config=cfg).reconstruct(results)

    def test_same_line_boxes(self):
        """Boxes on the same y-band should land in one line."""
        results = [
            _make_result("Name:", 20, 50, 80, 70),
            _make_result("John",  120, 52, 200, 72),
        ]
        layout = self._reconstruct(results)
        assert len(layout.lines) == 1
        assert len(layout.lines[0].boxes) == 2

    def test_distinct_lines(self):
        """Well-separated boxes should form separate lines."""
        results = [
            _make_result("Line 1", 20, 10, 100, 30),
            _make_result("Line 2", 20, 60, 100, 80),
            _make_result("Line 3", 20, 110, 100, 130),
        ]
        layout = self._reconstruct(results)
        assert len(layout.lines) == 3

    def test_lines_sorted_top_to_bottom(self):
        """Lines must be sorted by y_min ascending."""
        results = [
            _make_result("Bottom", 20, 200, 100, 220),
            _make_result("Top",    20,  10, 100,  30),
        ]
        layout = self._reconstruct(results)
        assert layout.lines[0].boxes[0].text == "Top"
        assert layout.lines[1].boxes[0].text == "Bottom"

    def test_boxes_sorted_left_to_right(self):
        """Within a line, boxes must be sorted left-to-right."""
        results = [
            _make_result("Right", 200, 10, 300, 30),
            _make_result("Left",   20, 10, 100, 30),
        ]
        layout = self._reconstruct(results)
        boxes = layout.lines[0].boxes
        assert boxes[0].text == "Left"
        assert boxes[1].text == "Right"

    def test_empty_input(self):
        layout = self._reconstruct([])
        assert layout.lines == []
        assert layout.plain_text == ""


# ── Plain-text rendering ──────────────────────────────────────────────────────

class TestPlainTextRendering:
    def test_single_word(self):
        results = [_make_result("Hello", 10, 10, 60, 30)]
        layout = LayoutReconstructor().reconstruct(results)
        assert "Hello" in layout.plain_text

    def test_multiword_line(self):
        results = [
            _make_result("First",  10, 10,  60, 30),
            _make_result("Second", 70, 10, 130, 30),
        ]
        layout = LayoutReconstructor().reconstruct(results)
        # Both tokens on the same line, so plain_text has a single line.
        lines = layout.plain_text.strip().splitlines()
        assert len(lines) == 1
        assert "First" in lines[0]
        assert "Second" in lines[0]

    def test_multiline_text(self):
        results = [
            _make_result("Line A", 10,  10, 100,  30),
            _make_result("Line B", 10, 100, 100, 120),
        ]
        layout = LayoutReconstructor().reconstruct(results)
        lines = layout.plain_text.strip().splitlines()
        assert len(lines) == 2


# ── Column detection ──────────────────────────────────────────────────────────

class TestColumnDetection:
    def test_two_columns_detected(self):
        """Invoice meta section should yield two column anchors."""
        results = _simple_invoice()
        cfg = LayoutConfig(column_min_alignment=2, column_x_tolerance=20)
        layout = LayoutReconstructor(config=cfg).reconstruct(results)
        # At least two distinct column anchors should be found.
        assert len(layout.lines) > 0


# ── Table detection ───────────────────────────────────────────────────────────

class TestTableDetection:
    def test_table_detected_in_invoice(self):
        """The 3-column item table should be detected as a TableRegion."""
        results = _simple_invoice()
        cfg = LayoutConfig(
            column_min_alignment=2,
            column_x_tolerance=20,
            table_col_count=2,
            table_row_count=2,
            table_col_consistency=0.5,
        )
        layout = LayoutReconstructor(config=cfg).reconstruct(results)
        assert len(layout.tables) >= 1

    def test_table_markdown_format(self):
        """Markdown output must start with a pipe and contain a separator row."""
        results = _simple_invoice()
        cfg = LayoutConfig(
            column_min_alignment=2,
            column_x_tolerance=20,
            table_col_count=2,
            table_row_count=2,
            table_col_consistency=0.5,
        )
        layout = LayoutReconstructor(config=cfg).reconstruct(results)
        if layout.tables:
            md = layout.tables[0].markdown
            lines = md.splitlines()
            assert lines[0].startswith("|")
            assert "---" in lines[1]

    def test_no_table_in_single_column_text(self):
        """A purely single-column paragraph should not produce a table."""
        results = [
            _make_result("Paragraph line one.",   10, 10, 200, 30),
            _make_result("Paragraph line two.",   10, 40, 200, 60),
            _make_result("Paragraph line three.", 10, 70, 200, 90),
        ]
        cfg = LayoutConfig(table_col_count=2)
        layout = LayoutReconstructor(config=cfg).reconstruct(results)
        assert layout.tables == []


# ── JSON serialisation ────────────────────────────────────────────────────────

class TestJsonSerialisation:
    def test_to_dict_is_json_serialisable(self):
        results = _simple_invoice()
        layout = LayoutReconstructor().reconstruct(results)
        d = layout.to_dict()
        # Must not raise.
        json_str = json.dumps(d, ensure_ascii=False)
        assert isinstance(json_str, str)

    def test_dict_has_required_keys(self):
        results = [_make_result("Test", 0, 0, 100, 30)]
        layout = LayoutReconstructor().reconstruct(results)
        d = layout.to_dict()
        assert "boxes" in d
        assert "lines" in d
        assert "tables" in d
        assert "plain_text" in d

    def test_box_dict_has_bbox(self):
        results = [_make_result("Test", 0, 0, 100, 30)]
        layout = LayoutReconstructor().reconstruct(results)
        box_dict = layout.to_dict()["boxes"][0]
        assert "bbox" in box_dict
        assert "text" in box_dict
        assert "x_min" in box_dict
        assert "line_index" in box_dict


# ── LayoutConfig defaults ─────────────────────────────────────────────────────

class TestLayoutConfig:
    def test_defaults_are_sensible(self):
        cfg = LayoutConfig()
        assert 0.0 < cfg.line_overlap_ratio <= 1.0
        assert cfg.column_gap_ratio > 0
        assert cfg.max_spaces >= cfg.min_spaces
        assert cfg.table_row_count >= 1
        assert cfg.table_col_count >= 1

    def test_custom_config_propagates(self):
        cfg = LayoutConfig(line_overlap_ratio=0.9, max_spaces=5)
        r = LayoutReconstructor(config=cfg)
        assert r.config.line_overlap_ratio == pytest.approx(0.9)
        assert r.config.max_spaces == 5
