"""
main.py – CLI entry point for the OCR pipeline.

Usage
-----
    python main.py --image path/to/image.jpg
    python main.py --image path/to/image.jpg --save_vis result/annotated.jpg
    python main.py --image path/to/image.jpg --device cuda:0 --min_det 0.6

Layout reconstruction
---------------------
    python main.py --image invoice.jpg --layout
    python main.py --image invoice.jpg --layout --json-out result/layout.json
"""

import argparse
import json
import logging
import sys

from src.ocr.pipeline   import OCRPipeline
from src.schemas.ocr_config import OCRConfig, DetectionConfig, RecognitionConfig
from src.layout.reconstructor import LayoutReconstructor
from src.schemas.layout_config import LayoutConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full OCR pipeline (detect + recognise) on an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--save_vis", "-v",
        default=None,
        help="If set, save an annotated visualisation to this path.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device for VietOCR: 'cpu' or 'cuda:0'.",
    )
    parser.add_argument(
        "--model",
        default="vgg_transformer",
        help="VietOCR model name.",
    )
    parser.add_argument(
        "--min_det",
        type=float,
        default=0.5,
        help="Minimum detection confidence threshold.",
    )
    parser.add_argument(
        "--min_rec",
        type=float,
        default=0.0,
        help="Minimum recognition confidence threshold.",
    )
    parser.add_argument(
        "--no_sort",
        action="store_true",
        help="Disable top-to-bottom, left-to-right sorting of boxes.",
    )
    parser.add_argument(
        "--layout", "-l",
        action="store_true",
        help="Run layout reconstruction and print structured output.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        metavar="PATH",
        help="If set, write the full layout JSON to this file path.",
    )
    parser.add_argument(
        "--line-overlap",
        type=float,
        default=0.5,
        help="line_overlap_ratio threshold for line grouping (0–1).",
    )
    parser.add_argument(
        "--col-gap",
        type=float,
        default=1.5,
        help="column_gap_ratio threshold for column detection.",
    )
    parser.add_argument(
        "--col-tol",
        type=int,
        default=15,
        help="column_x_tolerance in pixels for column anchor matching.",
    )
    parser.add_argument(
        "--verbose", "-V",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = OCRConfig(
        detection=DetectionConfig(min_score=args.min_det),
        recognition=RecognitionConfig(
            config_name=args.model,
            device=args.device,
            min_score=args.min_rec,
        ),
        sort_boxes=not args.no_sort,
    )

    pipeline = OCRPipeline(config=config)

    print(f"\nProcessing: {args.image}\n{'─' * 60}")
    results = pipeline.run(args.image, save_vis=args.save_vis)

    if not results:
        print("No text detected.")
        sys.exit(0)

    for idx, r in enumerate(results):
        det_score = f"{r['detection_score']:.3f}" if r["detection_score"] is not None else "N/A"
        rec_score = f"{r['recognition_score']:.3f}" if r["recognition_score"] is not None else "N/A"
        print(
            f"[{idx:03d}] det={det_score}  rec={rec_score}  "
            f"text=\"{r['text']}\""
        )

    print(f"\n{'─' * 60}")
    print(f"✅ Total regions: {len(results)}")

    if args.layout:
        _run_layout(results, args)
    else:
        print(f"\n📝 Full text:\n{pipeline.get_full_text(results)}")

    if args.save_vis:
        print(f"\n🖼️  Visualisation saved to: {args.save_vis}")


def _run_layout(results: list, args: argparse.Namespace) -> None:
    """Run layout reconstruction and print / save the structured output."""
    cfg = LayoutConfig(
        line_overlap_ratio=args.line_overlap,
        column_gap_ratio=args.col_gap,
        column_x_tolerance=args.col_tol,
    )
    reconstructor = LayoutReconstructor(config=cfg)
    layout = reconstructor.reconstruct(results)

    print(f"\n📐 Layout-preserved text:\n{'─' * 60}")
    print(layout.plain_text)

    if layout.tables:
        print(f"\n{'─' * 60}")
        print(f"📊 Detected {len(layout.tables)} table region(s):")
        for i, table in enumerate(layout.tables):
            print(f"\n── Table {i + 1} ({table.num_rows} rows × {table.num_cols} cols) ──")
            print(table.markdown)

    print(f"\n{'─' * 60}")
    print(f"📋 Lines: {len(layout.lines)}  |  Tables: {len(layout.tables)}")

    if args.json_out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(layout.to_dict(), fh, ensure_ascii=False, indent=2)
        print(f"\n💾 Layout JSON saved to: {args.json_out}")


if __name__ == "__main__":
    main()
