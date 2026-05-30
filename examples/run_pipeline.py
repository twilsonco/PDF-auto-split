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
    main()