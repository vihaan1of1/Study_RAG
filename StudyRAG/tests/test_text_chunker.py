"""Tests for Text Chunker.

Validates sliding window chunking, overlap, metadata provenance, and edge cases.
"""

import pytest
from src.pdf_processor import ProcessedDocument, ProcessedPage
from src.text_chunker import TextChunker


def test_chunk_splitting_and_overlap():
    """Verify that text is split into chunks of appropriate size with overlap."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=30)
    # 250 characters of sample sentences
    sample_text = (
        "Artificial Intelligence is transforming higher education. "
        "Retrieval-Augmented Generation bridges pre-trained models with domain knowledge. "
        "Dense vector indexes enable fast cosine similarity retrieval over academic papers."
    )
    page = ProcessedPage(page_number=3, text=sample_text, char_count=len(sample_text))

    chunks = chunker.chunk_page(doc_id="doc123", filename="notes.pdf", page=page)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.doc_id == "doc123"
        assert chunk.filename == "notes.pdf"
        assert chunk.page_number == 3
        assert chunk.char_count <= 110  # allow small tolerance for boundary finding

    # Check overlap: end of first chunk should have text appearing at start of second chunk
    first_end = chunks[0].text[-20:]
    assert any(first_end in c.text for c in chunks[1:])


def test_short_page_single_chunk():
    """Verify that pages shorter than chunk_size produce exactly one chunk."""
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    page = ProcessedPage(page_number=1, text="Short summary paragraph.", char_count=25)
    chunks = chunker.chunk_page(doc_id="d1", filename="lec.pdf", page=page)

    assert len(chunks) == 1
    assert chunks[0].text == "Short summary paragraph."
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_id == "d1_p1_c0"


def test_empty_page_handling():
    """Verify that an empty page yields zero chunks without error."""
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    page = ProcessedPage(page_number=2, text="   ", char_count=0)
    chunks = chunker.chunk_page(doc_id="d1", filename="lec.pdf", page=page)
    assert len(chunks) == 0


def test_chunk_document_metadata_provenance():
    """Verify that chunk_document preserves distinct page numbers across an entire document."""
    chunker = TextChunker(chunk_size=150, chunk_overlap=30)
    doc = ProcessedDocument(
        doc_id="doc_abc",
        filename="comp_arch.pdf",
        file_size_bytes=1000,
        total_pages=2,
        pages=[
            ProcessedPage(page_number=1, text="Page one discussing pipelining and hazards.", char_count=43),
            ProcessedPage(page_number=2, text="Page two discussing branch prediction and caches.", char_count=50),
        ],
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[0].filename == "comp_arch.pdf"
    assert chunks[1].page_number == 2
    assert chunks[1].filename == "comp_arch.pdf"
