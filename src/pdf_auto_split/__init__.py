#!/usr/bin/env python3
"""
Neuro-symbolic document processing pipeline.
Splits multi-document PDFs by combining fast text heuristics with vision LLM analysis.
"""

import argparse
import base64
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field
import fitz  # PyMuPDF
from openai import OpenAI

load_dotenv(find_dotenv())

DEFAULT_API_BASE = os.getenv("API_BASE", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-2.5-vision-72b")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
IMAGE_DPI = int(os.getenv("IMAGE_DPI", "150"))

FAST_PATH_TEXT_REGIONS_BOTTOM = (0.8, 1.0)
FAST_PATH_TEXT_REGIONS_TOP = (0.0, 0.2)

DEFAULT_VISION_PROMPT = """Analyze these two consecutive document pages and determine if they are part of the same continuous document.

Look for:
- Consistent headers, footers, page numbers
- Continued text flow across pages
- Similar formatting and layout
- Thematic continuity

Respond with JSON:
{
  "is_same_document": boolean,
  "confidence": integer (0-100),
  "reasoning": "string explaining the decision"
}"""


class VisionModelResponse(BaseModel):
    is_same_document: bool = Field(description="Whether the two pages are part of the same continuous document")
    confidence: int = Field(ge=0, le=100, description="Confidence score from 0-100")
    reasoning: str = Field(description="Explanation of the decision")


def parse_args():
    env_prompt = os.getenv("VISION_PROMPT", "").strip()
    default_prompt_display = (env_prompt[:50] + "...") if len(env_prompt) > 50 else env_prompt

    parser = argparse.ArgumentParser(
        description="Split a multi-document PDF using fast heuristics and vision LLM analysis."
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"Override API base URL (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=IMAGE_DPI,
        help=f"Image resolution for vision model (default: {IMAGE_DPI})",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=f"Vision model prompt (default: from .env or built-in)",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Load vision model prompt from file (overrides --prompt and .env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze PDF and report boundaries without splitting",
    )
    parser.add_argument(
        "input_pdf",
        help="Path to the PDF file to process",
    )
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def extract_region_text(page, y_start_ratio, y_end_ratio):
    """Extract text from a vertical region of the page."""
    clip = fitz.Rect(
        0,
        page.rect.height * y_start_ratio,
        page.rect.width,
        page.rect.height * y_end_ratio,
    )
    return page.get_text("text", clip=clip).strip()


def is_same_document_fast_path(page_n, page_np1) -> bool:
    """Returns True if text suggests same document (no LLM needed)."""
    bottom_text = extract_region_text(page_n, *FAST_PATH_TEXT_REGIONS_BOTTOM)
    top_text = extract_region_text(page_np1, *FAST_PATH_TEXT_REGIONS_TOP)

    if not bottom_text and not top_text:
        return False

    has_ending_punctuation = any(
        bottom_text.rstrip().endswith(p) for p in (".", "?", "!")
    )

    if has_ending_punctuation:
        return False

    top_text_stripped = top_text.lstrip()
    starts_lowercase = top_text_stripped and top_text_stripped[0].islower()
    common_continuations = ("the ", "a ", "an ", "and ", "or ", "but ", "to ", "of ", "in ", "on ", "that ", "which ", "who ", "whom ", "this ", "these ", "those ")
    starts_with_continuation = any(
        top_text_stripped.lower().startswith(c) for c in common_continuations
    )

    return starts_lowercase or starts_with_continuation


def render_page_to_base64_png(page, dpi) -> str:
    """Render a page to a base64-encoded PNG image."""
    scale = dpi / 72
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    return base64.b64encode(pixmap.tobytes("png")).decode("utf-8")


def call_vision_model(page_n, page_np1, api_base, dpi, prompt=None) -> VisionModelResponse:
    """Renders pages to images and calls vision model."""
    img_n = render_page_to_base64_png(page_n, dpi)
    img_np1 = render_page_to_base64_png(page_np1, dpi)

    api_key = os.getenv("API_KEY", "") or None
    client_kwargs = {"base_url": api_base, "timeout": API_TIMEOUT}
    if api_key:
        client_kwargs["api_key"] = api_key
    client = OpenAI(**client_kwargs)

    if prompt is None:
        env_prompt = os.getenv("VISION_PROMPT", "").strip()
        prompt = env_prompt if env_prompt else DEFAULT_VISION_PROMPT

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_n}"},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_np1}"},
                            },
                        ],
                    }
                ],
            )
            content = response.choices[0].message.content
            logging.debug(f"API response: {content}")

            content_clean = content.strip()
            if content_clean.startswith("```json"):
                content_clean = content_clean[7:]
            elif content_clean.startswith("```"):
                content_clean = content_clean[3:]
            if content_clean.endswith("```"):
                content_clean = content_clean[:-3]

            parsed = json.loads(content_clean.strip())
            return VisionModelResponse(**parsed)

        except (json.JSONDecodeError, Exception) as e:
            logging.warning(f"Malformed JSON response (attempt {attempt + 1}): {e}")
            if attempt == 1:
                logging.error("API call failed after retries: malformed JSON")
                return VisionModelResponse(
                    is_same_document=False,
                    confidence=0,
                    reasoning=f"API error: {e}",
                )

    return VisionModelResponse(
        is_same_document=False,
        confidence=0,
        reasoning="Max retries exceeded",
    )


def process_pdf(pdf_path, api_base, dpi, prompt=None):
    """Main processing loop returning list of boundary page numbers (1-indexed)."""
    doc = fitz.open(pdf_path)
    boundaries = []

    if len(doc) < 2:
        logging.info("PDF has fewer than 2 pages. Nothing to split.")
        return boundaries

    for i in range(len(doc) - 1):
        page_n = doc[i]
        page_np1 = doc[i + 1]

        if is_same_document_fast_path(page_n, page_np1):
            logging.info(f"Page pair ({i+1}, {i+2}) via Fast Path → Same document: True")
            continue

        logging.info(f"Page pair ({i+1}, {i+2}) via Slow Path")
        result = call_vision_model(page_n, page_np1, api_base, dpi, prompt=prompt)

        if not result.is_same_document:
            boundaries.append(i + 2)

        logging.info(
            f"Page pair ({i+1}, {i+2}) → Same document: {result.is_same_document} "
            f"(confidence: {result.confidence})"
        )

    return boundaries


def execute_split(pdf_path, boundaries):
    """Calls split_PDF.py with the detected boundaries."""
    cmd = ["python", "ref/split_PDF.py", pdf_path] + [str(b) for b in boundaries]
    logging.info(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    setup_logging()

    if not Path(args.input_pdf).exists():
        logging.error(f"Input PDF not found: {args.input_pdf}")
        sys.exit(1)

    prompt = None
    if args.prompt_file:
        prompt = args.prompt_file.read_text().strip()
        logging.info(f"Loaded custom prompt from {args.prompt_file}")
    elif args.prompt:
        prompt = args.prompt

    boundaries = process_pdf(args.input_pdf, args.api_base, args.dpi, prompt=prompt)

    if not boundaries:
        logging.info("No document boundaries detected. Nothing to split.")
        return

    if args.dry_run:
        logging.info(f"[DRY RUN] Would split at boundaries: {boundaries}")
    else:
        execute_split(args.input_pdf, boundaries)


if __name__ == "__main__":
    main()