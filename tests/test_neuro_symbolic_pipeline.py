import io
from unittest.mock import MagicMock, patch

import fitz
import pytest


class TestExtractRegionText:
    def test_extracts_middle_region(self):
        page = MagicMock()
        page.rect.height = 100
        page.get_text.return_value = "middle text"

        from pdf_auto_split import extract_region_text

        result = extract_region_text(page, 0.4, 0.6)
        clip_arg = page.get_text.call_args[1]["clip"]
        assert clip_arg.y0 == 40
        assert clip_arg.y1 == 60

    def test_extracts_bottom_region(self):
        page = MagicMock()
        page.rect.height = 100
        page.get_text.return_value = "bottom text"

        from pdf_auto_split import extract_region_text

        result = extract_region_text(page, 0.8, 1.0)
        clip_arg = page.get_text.call_args[1]["clip"]
        assert clip_arg.y0 == 80
        assert clip_arg.y1 == 100


class TestIsSameDocumentFastPath:
    def test_same_document_high_similarity(self):
        page_n = MagicMock()
        page_np1 = MagicMock()

        def mock_get_text(mode, clip=None):
            if mode == "text":
                return "This is the same document content that continues across pages."
            return ""

        page_n.get_text = mock_get_text
        page_np1.get_text = lambda mode, clip=None: "This is the same document content that continues across pages."

        with patch("pdf_auto_split.get_embedding_model") as mock_get_model:
            mock_model = MagicMock()
            import numpy as np
            embedding = np.array([0.1, 0.2, 0.3])
            mock_model.encode.return_value = np.array([embedding, embedding])
            mock_get_model.return_value = mock_model

            from pdf_auto_split import is_same_document_fast_path
            result = is_same_document_fast_path(page_n, page_np1)
            assert result is True

    def test_different_document_low_similarity(self):
        page_n = MagicMock()
        page_np1 = MagicMock()

        def mock_get_text(mode, clip=None):
            if mode == "text":
                return "This is one document about medical bills."
            return ""

        page_np1.get_text = lambda mode, clip=None: "Chapter 2 completely different content."

        with patch("pdf_auto_split.get_embedding_model") as mock_get_model:
            import numpy as np
            mock_model = MagicMock()
            emb1 = np.array([1.0, 0.0, 0.0])
            emb2 = np.array([0.0, 1.0, 0.0])
            mock_model.encode.return_value = np.array([emb1, emb2])
            mock_get_model.return_value = mock_model

            from pdf_auto_split import is_same_document_fast_path
            result = is_same_document_fast_path(page_n, page_np1)
            assert result is False

    def test_both_empty_returns_false(self):
        page_n = MagicMock()
        page_np1 = MagicMock()

        def mock_get_text(mode, clip=None):
            return ""

        page_n.get_text = mock_get_text
        page_np1.get_text = mock_get_text

        from pdf_auto_split import is_same_document_fast_path
        result = is_same_document_fast_path(page_n, page_np1)
        assert result is False


class TestRenderPageToBase64Png:
    def test_renders_page_to_base64(self):
        page = MagicMock()
        pixmap_mock = MagicMock()
        pixmap_mock.tobytes.return_value = b"png_data"
        page.get_pixmap.return_value = pixmap_mock

        from pdf_auto_split import render_page_to_base64_png

        result = render_page_to_base64_png(page, 150)
        assert isinstance(result, str)

    def test_uses_correct_scale_for_dpi(self):
        page = MagicMock()
        pixmap_mock = MagicMock()
        pixmap_mock.tobytes.return_value = b"png_data"
        page.get_pixmap.return_value = pixmap_mock

        from pdf_auto_split import render_page_to_base64_png, fitz

        result = render_page_to_base64_png(page, 150)
        matrix_arg = page.get_pixmap.call_args[1]["matrix"]
        assert matrix_arg.a == 150 / 72
        assert matrix_arg.d == 150 / 72


class TestComputeCosineSimilarity:
    def test_identical_vectors(self):
        from pdf_auto_split import compute_cosine_similarity
        result = compute_cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert result == 1.0

    def test_orthogonal_vectors(self):
        from pdf_auto_split import compute_cosine_similarity
        result = compute_cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert result == 0.0

    def test_opposite_vectors(self):
        from pdf_auto_split import compute_cosine_similarity
        result = compute_cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert result == -1.0

    def test_zero_vector(self):
        from pdf_auto_split import compute_cosine_similarity
        result = compute_cosine_similarity([0.0, 0.0], [1.0, 0.0])
        assert result == 0.0


class TestExtractPageText:
    def test_extracts_full_text(self):
        page = MagicMock()
        page.get_text.return_value = "  Hello World  \n\n"

        from pdf_auto_split import extract_page_text

        result = extract_page_text(page)
        assert result == "Hello World"

    def test_empty_returns_empty(self):
        page = MagicMock()
        page.get_text.return_value = ""

        from pdf_auto_split import extract_page_text

        result = extract_page_text(page)
        assert result == ""


class TestParseArgs:
    def test_default_api_base(self):
        from pdf_auto_split import parse_args, DEFAULT_API_BASE

        with patch("sys.argv", ["prog", "input.pdf"]):
            args = parse_args()
        assert args.api_base == DEFAULT_API_BASE

    def test_custom_api_base(self):
        from pdf_auto_split import parse_args

        with patch("sys.argv", ["prog", "--api-base", "http://custom:9000/v1", "input.pdf"]):
            args = parse_args()
        assert args.api_base == "http://custom:9000/v1"

    def test_custom_dpi(self):
        from pdf_auto_split import parse_args

        with patch("sys.argv", ["prog", "--dpi", "300", "input.pdf"]):
            args = parse_args()
        assert args.dpi == 300

    def test_default_dpi(self):
        from pdf_auto_split import parse_args, IMAGE_DPI

        with patch("sys.argv", ["prog", "input.pdf"]):
            args = parse_args()
        assert args.dpi == IMAGE_DPI


class TestExecuteSplit:
    @patch("pdf_auto_split.subprocess.run")
    def test_calls_split_pdf_with_boundaries(self, mock_run):
        from pdf_auto_split import execute_split

        execute_split("/path/to/file.pdf", [3, 8])

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args == ["python", "ref/split_PDF.py", "/path/to/file.pdf", "3", "8"]

    @patch("pdf_auto_split.subprocess.run")
    def test_logs_command(self, mock_run):
        from pdf_auto_split import execute_split
        import logging

        with patch("pdf_auto_split.logging") as mock_logging:
            execute_split("/path/to/file.pdf", [3, 8])
            mock_logging.info.assert_called()


class TestProcessPdf:
    def test_single_page_pdf_returns_empty_boundaries(self):
        from pdf_auto_split import process_pdf

        doc = fitz.open()
        doc.new_page()

        with patch("pdf_auto_split.fitz.open", return_value=doc):
            boundaries = process_pdf("/fake.pdf", "http://localhost:8000/v1", 150)

        assert boundaries == []

    def test_calls_fast_path_for_each_pair(self):
        from pdf_auto_split import process_pdf

        doc = fitz.open()
        for _ in range(3):
            page = doc.new_page()

        with patch("pdf_auto_split.fitz.open", return_value=doc):
            with patch("pdf_auto_split.is_same_document_fast_path", return_value=True) as mock_fp:
                boundaries = process_pdf("/fake.pdf", "http://localhost:8000/v1", 150)
                assert mock_fp.call_count == 1

    def test_slow_path_called_when_fast_path_fails(self):
        from pdf_auto_split import process_pdf, VisionModelResponse

        doc = fitz.open()
        for _ in range(2):
            page = doc.new_page()

        mock_response = VisionModelResponse(same_document_confidence=-10, reasoning="test")

        with patch("pdf_auto_split.fitz.open", return_value=doc):
            with patch("pdf_auto_split.is_same_document_fast_path", return_value=False):
                with patch("pdf_auto_split.call_vision_model") as mock_slow:
                    mock_slow.return_value = mock_response
                    boundaries = process_pdf("/fake.pdf", "http://localhost:8000/v1", 150)
                    mock_slow.assert_called_once()