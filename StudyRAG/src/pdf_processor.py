"""StudyRAG PDF Processing Module.

Extracts text from PDF documents using PyMuPDF (fitz), preserving page numbers,
detecting encrypted/corrupted/scanned files, and sanitizing extracted text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import pymupdf as fitz

from src.utils import compute_file_hash, sanitize_text


@dataclass
class ProcessedPage:
    """Representation of an extracted PDF page."""
    page_number: int  # 1-indexed
    text: str
    char_count: int


@dataclass
class ProcessedDocument:
    """Representation of an ingested PDF document."""
    doc_id: str
    filename: str
    file_size_bytes: int
    total_pages: int
    pages: List[ProcessedPage] = field(default_factory=list)
    full_text: str = ""
    is_encrypted: bool = False
    is_scanned: bool = False
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Return True if document was successfully processed with usable text."""
        return self.error is None and not self.is_scanned and len(self.pages) > 0


class PDFProcessor:
    """Handles PDF ingestion, validation, and text extraction."""

    # Minimum average characters per page to consider a document machine-readable
    MIN_TOTAL_CHARS_THRESHOLD = 50

    @classmethod
    def process_pdf(cls, file_bytes: bytes, filename: str) -> ProcessedDocument:
        """Validate and extract text from raw PDF bytes.

        Args:
            file_bytes: Raw binary content of the PDF.
            filename: Original name of the uploaded file.

        Returns:
            ProcessedDocument containing extracted pages and metadata.
        """
        file_size = len(file_bytes)
        doc_id = compute_file_hash(file_bytes)

        # 1. Check for empty bytes
        if file_size == 0:
            return ProcessedDocument(
                doc_id=doc_id,
                filename=filename,
                file_size_bytes=0,
                total_pages=0,
                error="Empty file: uploaded file contains 0 bytes.",
            )

        # 2. Check for PDF header signature
        if b"%PDF-" not in file_bytes[:1024]:
            return ProcessedDocument(
                doc_id=doc_id,
                filename=filename,
                file_size_bytes=file_size,
                total_pages=0,
                error="Invalid file format: file does not have a valid PDF header (%PDF-).",
            )

        # 3. Open with PyMuPDF
        try:
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            return ProcessedDocument(
                doc_id=doc_id,
                filename=filename,
                file_size_bytes=file_size,
                total_pages=0,
                error=f"Corrupted or invalid PDF: {str(exc)}",
            )

        # 4. Check encryption
        if pdf_doc.is_encrypted or pdf_doc.needs_pass:
            pdf_doc.close()
            return ProcessedDocument(
                doc_id=doc_id,
                filename=filename,
                file_size_bytes=file_size,
                total_pages=0,
                is_encrypted=True,
                error="Encrypted PDF: Password-protected files are not supported. Please remove the password and re-upload.",
            )

        total_pages = len(pdf_doc)
        if total_pages == 0:
            pdf_doc.close()
            return ProcessedDocument(
                doc_id=doc_id,
                filename=filename,
                file_size_bytes=file_size,
                total_pages=0,
                error="Empty PDF: Document has 0 pages.",
            )

        # 5. Extract text page-by-page
        pages: List[ProcessedPage] = []
        full_text_parts: List[str] = []

        try:
            for page_idx in range(total_pages):
                page = pdf_doc[page_idx]
                raw_text = page.get_text("text") or ""
                clean_text = sanitize_text(raw_text)

                page_num = page_idx + 1  # 1-indexed
                pages.append(
                    ProcessedPage(
                        page_number=page_num,
                        text=clean_text,
                        char_count=len(clean_text),
                    )
                )
                if clean_text:
                    full_text_parts.append(clean_text)
        except Exception as exc:
            pdf_doc.close()
            return ProcessedDocument(
                doc_id=doc_id,
                filename=filename,
                file_size_bytes=file_size,
                total_pages=total_pages,
                error=f"Error reading pages from PDF: {str(exc)}",
            )
        finally:
            pdf_doc.close()

        total_chars = sum(p.char_count for p in pages)
        is_scanned = total_chars < cls.MIN_TOTAL_CHARS_THRESHOLD

        error_msg = None
        if is_scanned:
            error_msg = (
                f"Scanned document detected: Extracted only {total_chars} characters across "
                f"{total_pages} page(s). This PDF appears to be image-based or scanned. "
                "StudyRAG requires machine-readable text (OCR can be applied prior to uploading)."
            )

        return ProcessedDocument(
            doc_id=doc_id,
            filename=filename,
            file_size_bytes=file_size,
            total_pages=total_pages,
            pages=pages,
            full_text="\n\n".join(full_text_parts),
            is_scanned=is_scanned,
            error=error_msg,
        )
