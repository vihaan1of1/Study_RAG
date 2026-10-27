"""Tests for StudyRAG LLM Client (Gemini & Groq).

Validates provider configuration detection, JSON extraction from markdown fences,
error translation, and mock completion generation.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.llm_client import LLMClient


def test_offline_provider_configuration():
    """Verify that offline provider is recognized and requires no external API key."""
    client = LLMClient(provider="offline")
    assert client.provider == "offline"
    assert not client.is_configured
    assert client.default_model == "local-extractive"


def test_gemini_provider_configuration():
    """Verify Gemini configuration initializes with valid key."""
    client = LLMClient(provider="gemini", api_key="test_dummy_gemini_key_123456789")
    assert client.provider == "gemini"
    assert client.is_configured
    assert "gemini" in client.default_model


def test_json_extractor_clean_json():
    """Verify extraction of pure JSON strings."""
    raw = '{"name": "StudyRAG", "status": "active"}'
    parsed = LLMClient._extract_json(raw)
    assert parsed == {"name": "StudyRAG", "status": "active"}


def test_json_extractor_markdown_fence():
    """Verify extraction of JSON wrapped in markdown code blocks."""
    raw = """
    Here is the requested output:
    ```json
    [
        {"id": 1, "topic": "FAISS"},
        {"id": 2, "topic": "Cosine Similarity"}
    ]
    ```
    Hope this helps!
    """
    parsed = LLMClient._extract_json(raw)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["topic"] == "FAISS"


def test_json_extractor_bracket_detection():
    """Verify fallback bracket detection when markdown tags are missing or malformed."""
    raw = 'Random commentary before {"key": "value"} random commentary after'
    parsed = LLMClient._extract_json(raw)
    assert parsed == {"key": "value"}


def test_gemini_completion_mock():
    """Verify Gemini chat completion formatting and output handling."""
    client = LLMClient(provider="gemini", api_key="test_dummy_gemini_key_123456789")

    mock_response = MagicMock()
    mock_response.text = "Retrieval-Augmented Generation combines retrieval with generative LLMs."

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    client._gemini_client = mock_client

    messages = [
        {"role": "system", "content": "You are an academic study assistant."},
        {"role": "user", "content": "What is RAG?"},
    ]

    answer = client.generate_chat_completion(messages=messages, model="gemini-3.8-flash")
    assert "Retrieval-Augmented Generation" in answer
    mock_client.models.generate_content.assert_called_once()
