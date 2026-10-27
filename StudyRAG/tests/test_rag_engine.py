"""Tests for Vector Store and RAG Engine.

Validates FAISS indexing, metadata search, document filtering, document removal,
grounding prompt generation, verifiable citations, and API error handling.
"""

from unittest.mock import MagicMock
import pytest

from src.pdf_processor import ProcessedDocument, ProcessedPage
from src.text_chunker import TextChunk
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine
from src.llm_client import LLMClient


@pytest.fixture
def populated_vector_store():
    """Fixture providing an indexed vector store with two documents."""
    store = VectorStore()
    store.clear()

    doc1 = ProcessedDocument(
        doc_id="doc1",
        filename="operating_systems.pdf",
        file_size_bytes=2048,
        total_pages=2,
        pages=[
            ProcessedPage(page_number=1, text="Virtual memory uses paging to isolate address spaces.", char_count=55),
            ProcessedPage(page_number=2, text="Deadlock prevention requires breaking Coffman conditions.", char_count=58),
        ],
    )
    chunks1 = [
        TextChunk("c1", "doc1", "operating_systems.pdf", 1, "Virtual memory uses paging to isolate address spaces.", 0, 55),
        TextChunk("c2", "doc1", "operating_systems.pdf", 2, "Deadlock prevention requires breaking Coffman conditions.", 1, 58),
    ]
    store.add_document(doc1, chunks1)

    doc2 = ProcessedDocument(
        doc_id="doc2",
        filename="database_systems.pdf",
        file_size_bytes=1024,
        total_pages=1,
        pages=[
            ProcessedPage(page_number=1, text="ACID properties ensure reliable database transactions.", char_count=56),
        ],
    )
    chunks2 = [
        TextChunk("c3", "doc2", "database_systems.pdf", 1, "ACID properties ensure reliable database transactions.", 0, 56),
    ]
    store.add_document(doc2, chunks2)

    return store


def test_vector_search_and_similarity(populated_vector_store):
    """Verify that vector search returns relevant chunks with score and metadata."""
    results = populated_vector_store.search("How does virtual memory work?", top_k=2)
    assert len(results) > 0
    top = results[0]
    assert top["doc_id"] == "doc1"
    assert "Virtual memory" in top["text"]
    assert top["page_number"] == 1
    assert "score" in top
    assert top["score"] > 0.0


def test_document_filtering(populated_vector_store):
    """Verify that doc_filter restricts search results strictly to allowed documents."""
    # Query about transactions, but filter ONLY to operating_systems (doc1)
    results = populated_vector_store.search("transactions", top_k=5, doc_filter=["doc1"])
    for r in results:
        assert r["doc_id"] == "doc1"
        assert r["filename"] == "operating_systems.pdf"


def test_document_removal_and_index_rebuild(populated_vector_store):
    """Verify that removing a document eliminates its chunks from future searches."""
    assert populated_vector_store.total_documents == 2
    assert populated_vector_store.total_chunks == 3

    # Remove doc1
    removed = populated_vector_store.remove_document("doc1")
    assert removed is True
    assert populated_vector_store.total_documents == 1
    assert populated_vector_store.total_chunks == 1

    # Search for virtual memory: should NOT return anything from doc1
    results = populated_vector_store.search("Virtual memory", top_k=5)
    for r in results:
        assert r["doc_id"] != "doc1"


def test_rag_engine_citation_generation(populated_vector_store):
    """Verify that RAGEngine constructs verifiable citations from actual retrieved chunks."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_chat_completion.return_value = (
        "Virtual memory uses paging for isolation [Source 1]."
    )

    engine = RAGEngine(vector_store=populated_vector_store, llm_client=mock_llm)
    response = engine.answer_question("Explain paging in virtual memory", top_k=2)

    assert response.has_context is True
    assert len(response.citations) > 0

    c1 = response.citations[0]
    assert c1.source_index == 1
    assert c1.filename == "operating_systems.pdf"
    assert c1.page_number == 1
    assert "Virtual memory" in c1.snippet

    # Verify LLM was called with the context
    mock_llm.generate_chat_completion.assert_called_once()
    called_messages = mock_llm.generate_chat_completion.call_args[1]["messages"]
    system_msg = called_messages[0]["content"]
    user_prompt = called_messages[1]["content"]

    assert "StudyRAG" in system_msg
    assert "DOCUMENT CONTEXT:" in user_prompt
    assert "operating_systems.pdf" in user_prompt


def test_rag_engine_empty_store():
    """Verify that querying an empty vector store gives a clean message without crashing."""
    empty_store = VectorStore()
    empty_store.clear()
    mock_llm = MagicMock(spec=LLMClient)

    engine = RAGEngine(vector_store=empty_store, llm_client=mock_llm)
    response = engine.answer_question("What is AI?")

    assert response.has_context is False
    assert "No documents have been indexed" in response.answer
    assert len(response.citations) == 0
    mock_llm.generate_chat_completion.assert_not_called()


def test_missing_api_key_handling():
    """Verify that LLMClient raises an informative ValueError when API key is missing."""
    client = LLMClient(api_key="")
    with pytest.raises(ValueError) as excinfo:
        client.generate_chat_completion([{"role": "user", "content": "hi"}])
    assert "Groq API key is not configured" in str(excinfo.value)
