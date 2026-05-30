#!/usr/bin/env python3
"""
Example script demonstrating the document processing pipeline.
Run with: uv run python examples/run_pipeline.py

All arguments passed after -- will be forwarded to the pipeline.
Example: uv run python examples/run_pipeline.py -- --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pdf_auto_split import main

DEMO_PDF = Path(__file__).parent / "demo.pdf"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run demo pipeline")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEMO_PDF,
        help=f"Path to demo PDF (default: {DEMO_PDF})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (analyze only, don't split)",
    )
    parser.add_argument(
        "--known-boundaries",
        type=str,
        default=None,
        help="Comma-separated known correct boundaries for accuracy assessment (e.g., '3,7,12')",
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Extra arguments to pass through (use -- to separate)",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"Error: Demo PDF not found at {args.pdf}")
        print("Please place a multi-page PDF at that location or specify another file with --pdf")
        sys.exit(1)

    cmd_args = ["run_pipeline.py", str(args.pdf)]
    if args.dry_run:
        cmd_args.append("--dry-run")
    cmd_args.extend(args.extra_args)

    sys.argv = cmd_args
    detected_boundaries = main()

    if args.known_boundaries and detected_boundaries is not None:
        known = set(int(x.strip()) for x in args.known_boundaries.split(","))
        detected = set(detected_boundaries)

        if not known and not detected:
            print("\n=== Accuracy Assessment ===")
            print("No boundaries in reference or results.")
            sys.exit(0)

        true_positives = len(known & detected)
        false_positives = len(detected - known)
        false_negatives = len(known - detected)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print("\n=== Accuracy Assessment ===")
        print(f"Known boundaries:     {sorted(known)}")
        print(f"Detected boundaries:  {sorted(detected)}")
        print(f"True positives:       {true_positives}")
        print(f"False positives:      {false_positives}")
        print(f"False negatives:      {false_negatives}")
        print(f"Precision:            {precision:.2%}")
        print(f"Recall:               {recall:.2%}")
        print(f"F1 Score:             {f1:.2%}")