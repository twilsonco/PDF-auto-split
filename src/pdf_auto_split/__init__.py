#!/usr/bin/env python3
"""
Neuro-symbolic document processing pipeline.
Splits multi-document PDFs by combining fast text heuristics with vision LLM analysis.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field
import fitz
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv(find_dotenv())

DEFAULT_API_BASE: str = os.getenv("API_BASE", "http://localhost:8000/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen-2.5-vision-72b")
API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "30"))
IMAGE_DPI: int = int(os.getenv("IMAGE_DPI", "150"))
CONTEXT_PAGES: int = int(os.getenv("CONTEXT_PAGES", "3"))

EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_API_BASE: str | None = os.getenv("EMBEDDING_API_BASE") or None
EMBEDDING_API_KEY: str | None = os.getenv("EMBEDDING_API_KEY") or None
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
HF_TOKEN: str | None = os.getenv("HF_TOKEN")  # Only used for local SentenceTransformer (rate limits with HF Hub)

_embedding_model: SentenceTransformer | None = None
_embedding_client: OpenAI | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, token=HF_TOKEN)
    return _embedding_model


def get_embedding_client() -> OpenAI:
    global _embedding_client
    if _embedding_client is None:
        client_kwargs: dict[str, Any] = {"base_url": EMBEDDING_API_BASE, "timeout": API_TIMEOUT}
        if EMBEDDING_API_KEY:
            client_kwargs["api_key"] = EMBEDDING_API_KEY
        _embedding_client = OpenAI(**client_kwargs)
    return _embedding_client


def get_embeddings(texts: list[str]) -> list[list[float]]:
    if EMBEDDING_API_BASE:
        client = get_embedding_client()
        response = client.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=texts,
        )
        return [item.embedding for item in response.data]
    else:
        model = get_embedding_model()
        return model.encode(texts, convert_to_numpy=True).tolist()


DEFAULT_VISION_PROMPT: str = """Analyze these document pages and determine if the last page is part of the same continuous document as the preceding pages.

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


def parse_args() -> argparse.Namespace:
    env_prompt = os.getenv("VISION_PROMPT", "").strip()
    if len(env_prompt) > 50:
        pass

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
        help="Vision model prompt (default: from .env or built-in)",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Load vision model prompt from file (overrides --prompt and .env)",
    )
    parser.add_argument(
        "--fast-path",
        action="store_true",
        help="Enable embedding-based fast path for page pair comparison. "
             "Not recommended for batches of similar documents (e.g., invoices).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze PDF and report boundaries without splitting"
    )
    parser.add_argument(
        "input_pdf",
        help="Path to the PDF file to process",
    )
    return parser.parse_args()


def setup_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "WARN").upper()
    numeric_level = getattr(logging, log_level, logging.WARN)
    logging.basicConfig(
        level=numeric_level,
        format="[%(levelname)s] %(message)s",
    )


def extract_region_text(page: fitz.Page, y_start_ratio: float, y_end_ratio: float) -> str:
    """Extract text from a vertical region of the page."""
    clip = fitz.Rect(
        0,
        page.rect.height * y_start_ratio,
        page.rect.width,
        page.rect.height * y_end_ratio,
    )
    return page.get_text("text", clip=clip).strip()


def extract_page_text(page: fitz.Page) -> str:
    """Extract full text content from a page."""
    return page.get_text("text").strip()


def compute_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def is_same_document_fast_path(page_n: fitz.Page, page_np1: fitz.Page) -> bool:
    """Returns True if embeddings suggest same document (no LLM needed)."""
    text_n = extract_page_text(page_n)
    text_np1 = extract_page_text(page_np1)
    
    logging.info(f"Fast path text lengths: page_n={len(text_n)}, page_np1={len(text_np1)}")

    if not text_n or not text_np1:
        return False
    
    logging.info("Computing embeddings for fast path...")

    embeddings = get_embeddings([text_n, text_np1])

    similarity = compute_cosine_similarity(embeddings[0], embeddings[1])
    logging.info(f"Embedding similarity: {similarity:.3f} (threshold: {SIMILARITY_THRESHOLD}; Same page: {similarity >= SIMILARITY_THRESHOLD})")

    return bool(similarity >= SIMILARITY_THRESHOLD)


def render_page_to_base64_png(page: fitz.Page, dpi: int) -> str:
    """Render a page to a base64-encoded PNG image."""
    scale = dpi / 72
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    return base64.b64encode(pixmap.tobytes("png")).decode("utf-8")


def call_vision_model(
    pages: list[fitz.Page],
    api_base: str,
    dpi: int,
    prompt: str | None = None,
) -> VisionModelResponse:
    """Renders pages to images and calls vision model."""
    context_count = len(pages) - 1

    api_key = os.getenv("API_KEY", "") or None
    client_kwargs: dict[str, Any] = {"base_url": api_base, "timeout": API_TIMEOUT}
    if api_key:
        client_kwargs["api_key"] = api_key
    client = OpenAI(**client_kwargs)

    if prompt is None:
        env_prompt = os.getenv("VISION_PROMPT", "").strip()
        prompt = env_prompt if env_prompt else DEFAULT_VISION_PROMPT

    final_prompt = re.sub(r'\{context_count\}', str(context_count), prompt)

    for attempt in range(2):
        try:
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": final_prompt}]
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


def process_pdf(
    pdf_path: str | Path,
    api_base: str,
    dpi: int,
    prompt: str | None = None,
    context_pages: int = CONTEXT_PAGES,
    use_fast_path: bool = False,
) -> list[int]:
    """Main processing loop returning list of boundary page numbers (1-indexed).

    Args:
        use_fast_path: Enable embedding-based fast path for page pair comparison.
            Useful when documents have substantial text content and are visually
            distinct. Not recommended for batches of similar documents (e.g.,
            multiple invoices from the same provider) where pages may differ only
            in small details like invoice numbers or dates.
    """
    doc = fitz.open(pdf_path)
    boundaries: list[int] = []

    if len(doc) < 2:
        logging.info("PDF has fewer than 2 pages. Nothing to split.")
        return boundaries

    window: list[int] = [0]

    for i in range(1, len(doc)):
        page_i = doc[i]
        window_size = len(window)

        should_use_fast_path = use_fast_path and window_size == 1 and context_pages >= 2
        if should_use_fast_path:
            page_prev = doc[window[0]]
            if is_same_document_fast_path(page_prev, page_i):
                logging.info(f"Page pair ({window[0]+1}, {i+1}) via Fast Path -> Same document: True")
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
                    f"Pages {window[0]+1}..{i+1} -> Same document confidence: {result.same_document_confidence}\n  Reasoning: {result.reasoning}"
                )
                window.append(i)
        else:
            pages_to_send = [doc[p] for p in window] + [page_i]
            logging.debug(f"Pages {window[0]+1}..{i+1} via Slow Path (context: {len(window)})")
            result = call_vision_model(pages_to_send, api_base, dpi, prompt=prompt)

            if result.same_document_confidence < 0:
                boundaries.append(i + 1)
                window = [i]
            else:
                logging.debug(
                    f"Pages {window[0]+1}..{i+1} -> Same document confidence: {result.same_document_confidence}\n  Reasoning: {result.reasoning}"
                )
                window.append(i)

    return boundaries


def execute_split(pdf_path: str | Path, boundaries: list[int]) -> None:
    """Calls split_PDF.py with the detected boundaries."""
    cmd: list[str] = ["python", "ref/split_PDF.py", str(pdf_path)] + [str(b) for b in boundaries]
    logging.info(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> list[int]:
    args = parse_args()
    setup_logging()

    if not Path(args.input_pdf).exists():
        logging.error(f"Input PDF not found: {args.input_pdf}")
        sys.exit(1)

    prompt: str | None = None
    if args.prompt_file:
        prompt = args.prompt_file.read_text().strip()
        logging.info(f"Loaded custom prompt from {args.prompt_file}")
    elif args.prompt:
        prompt = args.prompt

    boundaries = process_pdf(args.input_pdf, args.api_base, args.dpi, prompt=prompt, context_pages=args.context_pages, use_fast_path=args.fast_path)

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