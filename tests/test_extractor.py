"""Tests for text extractor service."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.extractor import TextExtractor


class TestTextExtractorInit:
    """Tests for TextExtractor initialization."""

    def test_default_whisper_model(self):
        """Should use default whisper model."""
        extractor = TextExtractor()
        assert extractor.whisper_model == "large"
        assert extractor._whisper is None

    def test_custom_whisper_model(self):
        """Should accept custom whisper model."""
        extractor = TextExtractor(whisper_model="small")
        assert extractor.whisper_model == "small"


class TestSupportedExtensions:
    """Tests for supported_extensions method."""

    def test_returns_set(self):
        """Should return a set of extensions."""
        extensions = TextExtractor.supported_extensions()
        assert isinstance(extensions, set)

    def test_includes_text_formats(self):
        """Should include text file formats."""
        extensions = TextExtractor.supported_extensions()
        assert ".txt" in extensions
        assert ".md" in extensions

    def test_includes_document_formats(self):
        """Should include document formats."""
        extensions = TextExtractor.supported_extensions()
        assert ".pdf" in extensions
        assert ".pptx" in extensions

    def test_includes_audio_formats(self):
        """Should include audio formats."""
        extensions = TextExtractor.supported_extensions()
        assert ".mp3" in extensions
        assert ".wav" in extensions
        assert ".m4a" in extensions


class TestExtractText:
    """Tests for text extraction methods."""

    @pytest.mark.asyncio
    async def test_extract_txt_file(self, tmp_path: Path):
        """Should extract text from .txt files."""
        text_file = tmp_path / "test.txt"
        text_file.write_text("Hello, World!", encoding="utf-8")

        extractor = TextExtractor()
        result = await extractor.extract(text_file)
        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_extract_md_file(self, tmp_path: Path):
        """Should extract text from .md files."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Title\n\nParagraph", encoding="utf-8")

        extractor = TextExtractor()
        result = await extractor.extract(md_file)
        assert "# Title" in result
        assert "Paragraph" in result

    @pytest.mark.asyncio
    async def test_extract_unsupported_raises(self, tmp_path: Path):
        """Should raise ValueError for unsupported formats."""
        unsupported = tmp_path / "test.xyz"
        unsupported.write_text("content", encoding="utf-8")

        extractor = TextExtractor()
        with pytest.raises(ValueError, match="Unsupported file format"):
            await extractor.extract(unsupported)

    @pytest.mark.asyncio
    async def test_extract_empty_file(self, tmp_path: Path):
        """Should handle empty files."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        extractor = TextExtractor()
        result = await extractor.extract(empty_file)
        assert result == ""


class TestPdfExtraction:
    """Tests for PDF extraction."""

    def test_extract_pdf_sync_callable(self):
        """Should have sync PDF extraction method."""
        extractor = TextExtractor()
        assert callable(extractor._extract_pdf_sync)

    def test_extract_pdf_sync_with_mock(self, tmp_path: Path):
        """Should extract text from PDF using pdfplumber."""
        import sys

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        mock_pdfplumber = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Extracted PDF text"
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=None)
        mock_pdfplumber.open.return_value = mock_pdf

        with patch.dict(sys.modules, {"pdfplumber": mock_pdfplumber}):
            extractor = TextExtractor()
            result = extractor._extract_pdf_sync(pdf_file)

            assert result == "Extracted PDF text"


class TestPptxExtraction:
    """Tests for PowerPoint extraction."""

    def test_extract_pptx_sync_callable(self):
        """Should have sync PPTX extraction method."""
        extractor = TextExtractor()
        assert callable(extractor._extract_pptx_sync)

    def test_extract_pptx_sync_with_mock(self, tmp_path: Path):
        """Should extract text from PowerPoint."""
        import sys

        pptx_file = tmp_path / "test.pptx"
        pptx_file.write_bytes(b"fake pptx content")

        mock_pptx = MagicMock()
        mock_slide = MagicMock()
        mock_shape = MagicMock()
        mock_shape.text = "Slide text"
        mock_slide.shapes = [mock_shape]
        mock_slide.has_notes_slide = False
        mock_pptx.Presentation.return_value.slides = [mock_slide]

        with patch.dict(sys.modules, {"pptx": mock_pptx}):
            extractor = TextExtractor()
            result = extractor._extract_pptx_sync(pptx_file)

            assert "Slide text" in result


class TestAudioTranscription:
    """Tests for audio transcription."""

    def test_transcribe_audio_sync_callable(self):
        """Should have sync audio transcription method."""
        extractor = TextExtractor()
        assert callable(extractor._transcribe_audio_sync)

    def test_whisper_lazy_loading(self):
        """Should not load whisper model until needed."""
        extractor = TextExtractor()
        assert extractor._whisper is None

    def test_transcribe_audio_loads_whisper(self, tmp_path: Path):
        """Should load whisper model when transcribing."""
        import sys

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio content")

        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Transcribed audio"}
        mock_whisper.load_model.return_value = mock_model

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            extractor = TextExtractor()
            result = extractor._transcribe_audio_sync(audio_file)

            assert result == "Transcribed audio"
            mock_whisper.load_model.assert_called_once_with("large")


class TestExtractAsync:
    """Tests for async extraction wrapper."""

    @pytest.mark.asyncio
    async def test_extract_pdf_async(self, tmp_path: Path):
        """Should run PDF extraction in executor."""
        import sys

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf")

        mock_pdfplumber = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF text"
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=None)
        mock_pdfplumber.open.return_value = mock_pdf

        with patch.dict(sys.modules, {"pdfplumber": mock_pdfplumber}):
            extractor = TextExtractor()
            result = await extractor.extract(pdf_file)

            assert result == "PDF text"

    @pytest.mark.asyncio
    async def test_extract_pptx_async(self, tmp_path: Path):
        """Should run PPTX extraction in executor."""
        import sys

        pptx_file = tmp_path / "test.pptx"
        pptx_file.write_bytes(b"fake pptx")

        mock_pptx = MagicMock()
        mock_pptx.Presentation.return_value.slides = []

        with patch.dict(sys.modules, {"pptx": mock_pptx}):
            extractor = TextExtractor()
            result = await extractor.extract(pptx_file)

            assert result == ""

    @pytest.mark.asyncio
    async def test_extract_audio_async(self, tmp_path: Path):
        """Should run audio transcription in executor."""
        import sys

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Audio text"}
        mock_whisper.load_model.return_value = mock_model

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            extractor = TextExtractor()
            result = await extractor.extract(audio_file)

            assert result == "Audio text"
