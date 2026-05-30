import io
from unittest.mock import MagicMock, patch

import fitz
import pytest


class TestExtractRegionText:
    def test_extracts_middle_region(self):
        page = MagicMock()
        page.rect.height = 100
        page.get_text.return_value = "middle text"

        from file_organization import extract_region_text

        result = extract_region_text(page, 0.4, 0.6)
        clip_arg = page.get_text.call_args[1]["clip"]
        assert clip_arg.y0 == 40
        assert clip_arg.y1 == 60

    def test_extracts_bottom_region(self):
        page = MagicMock()
        page.rect.height = 100
        page.get_text.return_value = "bottom text"

        from file_organization import extract_region_text

        result = extract_region_text(page, 0.8, 1.0)
        clip_arg = page.get_text.call_args[1]["clip"]
        assert clip_arg.y0 == 80
        assert clip_arg.y1 == 100


class TestIsSameDocumentFastPath:
    def test_same_document_continuation_lowercase(self):
        page_n = MagicMock()
        page_np1 = MagicMock()

        def mock_get_text(mode, clip=None):
            if mode == "text":
                if clip and clip.y0 > 70:
                    return "and the next"
            return ""

        page_n.get_text = mock_get_text
        page_np1.get_text = lambda mode, clip=None: "the story continues"

        from file_organization import is_same_document_fast_path

        result = is_same_document_fast_path(page_n, page_np1)
        assert result is True

    def test_different_document_ends_with_period(self):
        page_n = MagicMock()
        page_np1 = MagicMock()

        def mock_get_text(mode, clip=None):
            if mode == "text":
                if clip and clip.y0 > 70:
                    return "The document ends here."
            return ""

        page_n.get_text = mock_get_text
        page_np1.get_text = lambda mode, clip=None: "Chapter 2"

        from file_organization import is_same_document_fast_path

        result = is_same_document_fast_path(page_n, page_np1)
        assert result is False

    def test_different_document_starts_with_uppercase(self):
        page_n = MagicMock()
        page_np1 = MagicMock()

        def mock_get_text(mode, clip=None):
            if mode == "text":
                if clip and clip.y0 > 70:
                    return "continues here"
            return ""

        page_n.get_text = mock_get_text
        page_np1.get_text = lambda mode, clip=None: "Chapter 2"

        from file_organization import is_same_document_fast_path

        result = is_same_document_fast_path(page_n, page_np1)
        assert result is False

    def test_both_empty_returns_false(self):
        page_n = MagicMock()
        page_np1 = MagicMock()

        def mock_get_text(mode, clip=None):
            return ""

        page_n.get_text = mock_get_text
        page_np1.get_text = mock_get_text

        from file_organization import is_same_document_fast_path

        result = is_same_document_fast_path(page_n, page_np1)
        assert result is False


class TestRenderPageToBase64Png:
    def test_renders_page_to_base64(self):
        page = MagicMock()
        pixmap_mock = MagicMock()
        pixmap_mock.tobytes.return_value = b"png_data"
        page.get_pixmap.return_value = pixmap_mock

        from file_organization import render_page_to_base64_png

        result = render_page_to_base64_png(page, 150)
        assert isinstance(result, str)

    def test_uses_correct_scale_for_dpi(self):
        page = MagicMock()
        pixmap_mock = MagicMock()
        pixmap_mock.tobytes.return_value = b"png_data"
        page.get_pixmap.return_value = pixmap_mock

        from file_organization import render_page_to_base64_png, fitz

        result = render_page_to_base64_png(page, 150)
        matrix_arg = page.get_pixmap.call_args[1]["matrix"]
        assert matrix_arg.a == 150 / 72
        assert matrix_arg.d == 150 / 72


class TestParseArgs:
    def test_default_api_base(self):
        from file_organization import parse_args, DEFAULT_API_BASE

        with patch("sys.argv", ["prog", "input.pdf"]):
            args = parse_args()
        assert args.api_base == DEFAULT_API_BASE

    def test_custom_api_base(self):
        from file_organization import parse_args

        with patch("sys.argv", ["prog", "--api-base", "http://custom:9000/v1", "input.pdf"]):
            args = parse_args()
        assert args.api_base == "http://custom:9000/v1"

    def test_custom_dpi(self):
        from file_organization import parse_args

        with patch("sys.argv", ["prog", "--dpi", "300", "input.pdf"]):
            args = parse_args()
        assert args.dpi == 300

    def test_default_dpi(self):
        from file_organization import parse_args, IMAGE_DPI

        with patch("sys.argv", ["prog", "input.pdf"]):
            args = parse_args()
        assert args.dpi == IMAGE_DPI


class TestExecuteSplit:
    @patch("file_organization.subprocess.run")
    def test_calls_split_pdf_with_boundaries(self, mock_run):
        from file_organization import execute_split

        execute_split("/path/to/file.pdf", [3, 8])

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args == ["python", "ref/split_PDF.py", "/path/to/file.pdf", "3", "8"]

    @patch("file_organization.subprocess.run")
    def test_logs_command(self, mock_run):
        from file_organization import execute_split
        import logging

        with patch("file_organization.logging") as mock_logging:
            execute_split("/path/to/file.pdf", [3, 8])
            mock_logging.info.assert_called()


class TestProcessPdf:
    def test_single_page_pdf_returns_empty_boundaries(self):
        from file_organization import process_pdf

        doc = fitz.open()
        doc.new_page()

        with patch("file_organization.fitz.open", return_value=doc):
            boundaries = process_pdf("/fake.pdf", "http://localhost:8000/v1", 150)

        assert boundaries == []

    def test_calls_fast_path_for_each_pair(self):
        from file_organization import process_pdf

        doc = fitz.open()
        for _ in range(3):
            page = doc.new_page()

        with patch("file_organization.fitz.open", return_value=doc):
            with patch("file_organization.is_same_document_fast_path", return_value=True) as mock_fp:
                boundaries = process_pdf("/fake.pdf", "http://localhost:8000/v1", 150)
                assert mock_fp.call_count == 2

    def test_slow_path_called_when_fast_path_fails(self):
        from file_organization import process_pdf, VisionModelResponse

        doc = fitz.open()
        for _ in range(2):
            page = doc.new_page()

        mock_response = VisionModelResponse(is_same_document=False, confidence=90, reasoning="test")

        with patch("file_organization.fitz.open", return_value=doc):
            with patch("file_organization.is_same_document_fast_path", return_value=False):
                with patch("file_organization.call_vision_model") as mock_slow:
                    mock_slow.return_value = mock_response
                    boundaries = process_pdf("/fake.pdf", "http://localhost:8000/v1", 150)
                    mock_slow.assert_called_once()