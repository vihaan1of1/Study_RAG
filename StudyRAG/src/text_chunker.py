"""StudyRAG Text Chunker Module.

Splits document text into overlapping, semantically coherent chunks while strictly
preserving document and page-level metadata for accurate RAG citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from src.pdf_processor import ProcessedDocument, ProcessedPage


@dataclass
class TextChunk:
    """A single chunk of text with complete provenance metadata."""
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    text: str
    chunk_index: int
    char_count: int

    def to_dict(self) -> dict:
        """Convert chunk metadata to dictionary for serialization / FAISS mapping."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
        }


class TextChunker:
    """Recursive/sliding window text chunker preserving page boundaries."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_page(
        self,
        doc_id: str,
        filename: str,
        page: ProcessedPage,
        starting_chunk_idx: int = 0,
    ) -> List[TextChunk]:
        """Split a single page's text into overlapping chunks.

        Args:
            doc_id: Unique document identifier.
            filename: Source PDF filename.
            page: ProcessedPage containing page number and text.
            starting_chunk_idx: Initial chunk index counter.

        Returns:
            List of TextChunk instances.
        """
        text = page.text.strip()
        if not text:
            return []

        chunks: List[TextChunk] = []
        text_len = len(text)

        # If text fits into a single chunk, emit immediately
        if text_len <= self.chunk_size:
            chunk_id = f"{doc_id}_p{page.page_number}_c{starting_chunk_idx}"
            return [
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page.page_number,
                    text=text,
                    chunk_index=starting_chunk_idx,
                    char_count=text_len,
                )
            ]

        start = 0
        local_idx = starting_chunk_idx

        while start < text_len:
            end = start + self.chunk_size

            if end >= text_len:
                chunk_str = text[start:].strip()
                if chunk_str:
                    chunk_id = f"{doc_id}_p{page.page_number}_c{local_idx}"
                    chunks.append(
                        TextChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            filename=filename,
                            page_number=page.page_number,
                            text=chunk_str,
                            chunk_index=local_idx,
                            char_count=len(chunk_str),
                        )
                    )
                break

            # Attempt to split at natural boundaries within a search window at the end
            sub_window = text[start:end]
            split_at = -1

            # Priority 1: Paragraph break (\n\n)
            last_para = sub_window.rfind("\n\n")
            if last_para > int(self.chunk_size * 0.6):
                split_at = last_para + 2
            else:
                # Priority 2: Sentence boundary (. or ? or ! followed by space or newline)
                sentence_match = list(re.finditer(r"[.!?]\s+", sub_window))
                if sentence_match:
                    last_sent = sentence_match[-1]
                    if last_sent.end() > int(self.chunk_size * 0.5):
                        split_at = last_sent.end()

            # Priority 3: Word boundary (whitespace)
            if split_at == -1:
                last_space = sub_window.rfind(" ")
                if last_space > int(self.chunk_size * 0.4):
                    split_at = last_space + 1

            # Fallback: Hard cut at chunk_size
            if split_at == -1:
                split_at = self.chunk_size

            chunk_str = text[start : start + split_at].strip()
            if chunk_str:
                chunk_id = f"{doc_id}_p{page.page_number}_c{local_idx}"
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        filename=filename,
                        page_number=page.page_number,
                        text=chunk_str,
                        chunk_index=local_idx,
                        char_count=len(chunk_str),
                    )
                )
                local_idx += 1

            # Advance start pointer with overlap
            new_start = start + split_at - self.chunk_overlap
            if new_start <= start:
                new_start = start + split_at  # Ensure forward progress
            start = new_start

        return chunks

    def chunk_document(self, doc: ProcessedDocument) -> List[TextChunk]:
        """Split an entire document into chunks, preserving page numbers."""
        if not doc.pages or not doc.is_valid:
            return []

        all_chunks: List[TextChunk] = []
        global_idx = 0

        for page in doc.pages:
            page_chunks = self.chunk_page(
                doc_id=doc.doc_id,
                filename=doc.filename,
                page=page,
                starting_chunk_idx=global_idx,
            )
            all_chunks.extend(page_chunks)
            global_idx += len(page_chunks)

        return all_chunks
