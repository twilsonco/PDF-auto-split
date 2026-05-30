#!/usr/bin/env python3
"""
Neuro-symbolic document processing pipeline.
Splits multi-document PDFs by combining fast text heuristics with vision LLM analysis.
"""

import argparse
import base64
import json
import logging
import re
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
CONTEXT_PAGES = int(os.getenv("CONTEXT_PAGES", "3"))

FAST_PATH_TEXT_REGIONS_BOTTOM = (0.8, 1.0)
FAST_PATH_TEXT_REGIONS_TOP = (0.0, 0.2)

DEFAULT_VISION_PROMPT = """Analyze these document pages and determine if the last page is part of the same continuous document as the preceding pages.

The first {context_count} pages are believed to be from the same document. Is the last page part of the same document?

Look for:
- Consistent headers, footers, page numbers
- Continued text flow across pages
- Similar formatting and layout
- Thematic continuity

Respond with JSON:
{
  "same_document_confidence": integer (-100 to +100),
  "reasoning": "string explaining the decision"
}"""


class VisionModelResponse(BaseModel):
    same_document_confidence: int = Field(
        ge=-100, le=100,
        description="Signed confidence from -100 (definitely different) to +100 (definitely same)"
    )
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
        "--context-pages",
        type=int,
        default=CONTEXT_PAGES,
        help=f"Number of consecutive pages to show when asking about continuity (default: {CONTEXT_PAGES})",
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


def call_vision_model(pages, api_base, dpi, prompt=None) -> VisionModelResponse:
    """Renders pages to images and calls vision model."""
    context_count = len(pages) - 1

    api_key = os.getenv("API_KEY", "") or None
    client_kwargs = {"base_url": api_base, "timeout": API_TIMEOUT}
    if api_key:
        client_kwargs["api_key"] = api_key
    client = OpenAI(**client_kwargs)

    if prompt is None:
        env_prompt = os.getenv("VISION_PROMPT", "").strip()
        prompt = env_prompt if env_prompt else DEFAULT_VISION_PROMPT

    # Use regex to replace only the exact {context_count} placeholder, avoiding issues
    # with user's prompt containing JSON or other curly braces
    final_prompt = re.sub(r'\{context_count\}', str(context_count), prompt)

    for attempt in range(2):
        try:
            content_parts = [{"type": "text", "text": final_prompt}]
            for page in pages:
                img_b64 = render_page_to_base64_png(page, dpi)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": content_parts,
                    }
                ],
            )
            content = response.choices[0].message.content
            logging.debug(f"API response: {content}")

            if content is None or content.strip() == "":
                logging.warning(f"Empty response from API on attempt {attempt + 1}")
                continue

            content_clean = content.strip()
            
            json_match = re.search(r'```json\s*(\{[^}]+\})\s*```', content_clean, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                content_clean = re.sub(r'^```json\s*', '', content_clean)
                content_clean = re.sub(r'\s*```$', '', content_clean)
                parsed = json.loads(content_clean.strip())

            return VisionModelResponse(**parsed)

        except (json.JSONDecodeError, Exception) as e:
            logging.warning(f"Malformed JSON response (attempt {attempt + 1}): {e}")
            if attempt == 1:
                logging.error("API call failed after retries: malformed JSON")
                return VisionModelResponse(
                    same_document_confidence=-100,
                    reasoning=f"API error: {e}",
                )

    return VisionModelResponse(
        same_document_confidence=-100,
        reasoning="Max retries exceeded",
    )


def process_pdf(pdf_path, api_base, dpi, prompt=None, context_pages=CONTEXT_PAGES):
    """Main processing loop returning list of boundary page numbers (1-indexed)."""
    doc = fitz.open(pdf_path)
    boundaries = []

    if len(doc) < 2:
        logging.info("PDF has fewer than 2 pages. Nothing to split.")
        return boundaries

    context_pages = context_pages
    window = [0]

    for i in range(1, len(doc)):
        page_i = doc[i]
        window_size = len(window)

        should_use_fast_path = (window_size == 1 and context_pages >= 2)
        if should_use_fast_path:
            page_prev = doc[window[0]]
            if is_same_document_fast_path(page_prev, page_i):
                logging.info(f"Page pair ({window[0]+1}, {i+1}) via Fast Path → Same document: True")
                window.append(i)
                continue

        if window_size >= context_pages:
            candidate_page = doc[window[-1]]
            pages_to_send = [doc[p] for p in window[:-1]] + [candidate_page]
            logging.info(f"Pages {window[0]+1}..{i+1} via Slow Path (context: {len(window)-1})")
            result = call_vision_model(pages_to_send, api_base, dpi, prompt=prompt)

            if result.same_document_confidence < 0:
                boundaries.append(i + 1)
                window = [i]
            else:
                logging.info(
                    f"Pages {window[0]+1}..{i+1} → Same document confidence: {result.same_document_confidence}\n  Reasoning: {result.reasoning}"
                )
                window.append(i)
        else:
            pages_to_send = [doc[p] for p in window] + [page_i]
            logging.info(f"Pages {window[0]+1}..{i+1} via Slow Path (context: {len(window)})")
            result = call_vision_model(pages_to_send, api_base, dpi, prompt=prompt)

            if result.same_document_confidence < 0:
                boundaries.append(i + 1)
                window = [i]
            else:
                logging.info(
                    f"Pages {window[0]+1}..{i+1} → Same document confidence: {result.same_document_confidence}\n  Reasoning: {result.reasoning}"
                )
                window.append(i)

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

    boundaries = process_pdf(args.input_pdf, args.api_base, args.dpi, prompt=prompt, context_pages=args.context_pages)

    if not boundaries:
        logging.info("No document boundaries detected. Nothing to split.")
        return boundaries

    if args.dry_run:
        logging.info(f"[DRY RUN] Would split at boundaries: {boundaries}")
    else:
        execute_split(args.input_pdf, boundaries)

    return boundaries


if __name__ == "__main__":
    main()