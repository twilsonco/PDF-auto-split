#!/usr/bin/env python3
"""
Example script demonstrating the document processing pipeline.
Run with: uv run python examples/run_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from file_organization import main

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
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"Error: Demo PDF not found at {args.pdf}")
        print("Please place a multi-page PDF at that location or specify another file with --pdf")
        sys.exit(1)

    sys.argv = ["run_pipeline.py", str(args.pdf)]
    main()