"""End-to-End Pipeline Integration Test.

Tests ingestion of a real generated academic PDF, chunking, FAISS embedding,
exact page-level retrieval, and similarity score validity.
"""

from pathlib import Path
import pytest
from src.pdf_processor import PDFProcessor
from src.text_chunker import TextChunker
from src.vector_store import VectorStore


def test_end_to_end_sample_pdf_ingestion():
    """Verify that sample academic PDF is parsed, indexed, and accurately retrieved."""
    pdf_path = Path("sample_docs/lecture_notes_rag.pdf")
    assert pdf_path.exists(), "Sample PDF must exist"

    # 1. PDF Processing
    pdf_bytes = pdf_path.read_bytes()
    doc = PDFProcessor.process_pdf(pdf_bytes, pdf_path.name)

    assert doc.is_valid is True
    assert doc.total_pages == 3
    assert len(doc.pages) == 3

    # Check page-level content
    assert "INTRODUCTION TO RAG" in doc.pages[0].text
    assert "COSINE SIMILARITY" in doc.pages[1].text
    assert "FAISS VECTOR INDEX" in doc.pages[2].text

    # 2. Text Chunking
    chunker = TextChunker(chunk_size=800, chunk_overlap=150)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 3

    # Every chunk must preserve page numbers 1, 2, or 3
    page_numbers = {c.page_number for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers
    assert 3 in page_numbers

    # 3. Vector Indexing
    store = VectorStore()
    num_indexed = store.add_document(doc, chunks)
    assert num_indexed == len(chunks)
    assert store.total_documents == 1
    assert store.total_chunks == len(chunks)

    # 4. Semantic Retrieval: Query about Cosine Similarity (from Page 2)
    results = store.search("What is the mathematical formulation of cosine similarity?", top_k=3)
    assert len(results) > 0

    top_result = results[0]
    assert top_result["page_number"] == 2
    assert "CosineSimilarity(q, d)" in top_result["text"]
    assert top_result["filename"] == "lecture_notes_rag.pdf"
    assert top_result["score"] > 0.5  # High semantic similarity

    # 5. Semantic Retrieval: Query about IndexFlatIP (from Page 3)
    results_faiss = store.search("Explain IndexFlatIP in FAISS", top_k=2)
    assert len(results_faiss) > 0
    top_faiss = results_faiss[0]
    assert top_faiss["page_number"] == 3
    assert "IndexFlatIP" in top_faiss["text"]
