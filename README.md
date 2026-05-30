# Neuro-Symbolic Document Processing Pipeline

Splits multi-document scanned PDFs into individual source documents using a two-tier approach:

1. **Fast Path**: Deterministic text-based heuristic (skips LLM for clear cases)
2. **Slow Path**: Vision LLM check (for ambiguous/empty text scenarios)

## Setup

```bash
uv sync
```

This creates a `.venv` and installs dependencies (`pymupdf`, `openai`, `pytest`).

## Running

```bash
uv run python -m file_organization <input_pdf> [--api-base URL] [--dpi N]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input_pdf` | Path to the PDF file to process | Required |
| `--api-base` | Override default API base URL | `http://localhost:8000/v1` |
| `--dpi` | Image resolution for vision model | `150` |

### Example

```bash
# Using default API endpoint
uv run python -m file_organization document.pdf

# Custom API and higher resolution
uv run python -m file_organization document.pdf --api-base http://localhost:9000/v1 --dpi 300
```

## How It Works

```
Input PDF → Page Iterator
              ↓
       [Fast Path Check]
       - Extract bottom 20% text from page N
       - Extract top 20% text from page N+1
       - Heuristic: mid-sentence continuation?
              ↓
  ┌───────────┴───────────┐
  │                       │
Pass                    Fail/Unclear
  ↓                        ↓
Next Pair       [Slow Path - Vision LLM]
              - Render both pages as images (base64)
              - Call local vision model
              - Parse JSON response {is_same_document, confidence}
                   ↓
          Record boundary if is_same_document=False
              ↓
       Aggregate All Boundaries → subprocess call to split_PDF.py
```

## Output

The script calls `ref/split_PDF.py` with detected boundaries, producing output files like:

```
document.pages001_002.pdf
document.pages003_007.pdf
document.pages008_015.pdf
```

## Testing

```bash
uv run pytest tests/ -v
```