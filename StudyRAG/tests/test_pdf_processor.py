"""Tests for PDF Processor.

Validates extraction, page preservation, scanned document detection, and error handling.
"""

import pymupdf as fitz
import pytest
from src.pdf_processor import PDFProcessor


def create_sample_pdf(pages_text: list[str]) -> bytes:
    """Helper to construct a valid in-memory PDF with specified pages."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 50), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_pdf_extraction_and_page_preservation():
    """Verify that multiple pages are extracted with correct 1-indexed page numbers."""
    page1_text = "Chapter 1: Introduction to Retrieval-Augmented Generation."
    page2_text = "Chapter 2: Vector Embeddings and Dense Retrieval Mechanisms."
    pdf_bytes = create_sample_pdf([page1_text, page2_text])

    doc = PDFProcessor.process_pdf(pdf_bytes, "test_lecture.pdf")

    assert doc.is_valid is True
    assert doc.error is None
    assert doc.total_pages == 2
    assert len(doc.pages) == 2

    assert doc.pages[0].page_number == 1
    assert "Retrieval-Augmented Generation" in doc.pages[0].text

    assert doc.pages[1].page_number == 2
    assert "Vector Embeddings" in doc.pages[1].text


def test_empty_file_handling():
    """Verify that zero-byte files are rejected gracefully."""
    doc = PDFProcessor.process_pdf(b"", "empty.pdf")
    assert doc.is_valid is False
    assert "0 bytes" in doc.error


def test_invalid_header_handling():
    """Verify that non-PDF files are detected and rejected."""
    invalid_bytes = b"This is plain text pretending to be a PDF file."
    doc = PDFProcessor.process_pdf(invalid_bytes, "fake.pdf")
    assert doc.is_valid is False
    assert "not have a valid PDF header" in doc.error


def test_scanned_pdf_detection():
    """Verify that PDFs with zero/minimal extractable text are flagged as scanned."""
    # Create an empty page PDF without text
    doc_fitz = fitz.open()
    doc_fitz.new_page()
    pdf_bytes = doc_fitz.tobytes()
    doc_fitz.close()

    doc = PDFProcessor.process_pdf(pdf_bytes, "scanned_doc.pdf")
    assert doc.is_scanned is True
    assert doc.is_valid is False
    assert "Scanned document detected" in doc.error
    assert "OCR" in doc.error
