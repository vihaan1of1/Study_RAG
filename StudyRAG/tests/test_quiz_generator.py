"""Tests for Quiz Generator.

Validates MCQ schema validation, option letter mapping, malformed JSON recovery,
duplicate deduplication, and accurate grading calculation.
"""

import pytest
from src.quiz_generator import QuizGenerator, QuizQuestion


def test_valid_quiz_validation():
    """Verify that compliant JSON array produces clean QuizQuestion objects."""
    raw_json = [
        {
            "id": 1,
            "question": "What is the primary function of a vector embedding in RAG?",
            "options": [
                "Map semantic meaning into high-dimensional geometric space",
                "Encrypt text for safe storage",
                "Compress PDFs into zip files",
                "Execute SQL queries",
            ],
            "correct_answer": "Map semantic meaning into high-dimensional geometric space",
            "explanation": "Embeddings represent semantic similarities geometrically.",
            "source": "rag_intro.pdf, Page 4",
        }
    ]

    questions = QuizGenerator._validate_and_sanitize_quiz(raw_json, expected_count=1)
    assert len(questions) == 1
    q = questions[0]
    assert q.id == 1
    assert "vector embedding" in q.question
    assert len(q.options) == 4
    assert q.correct_answer == "Map semantic meaning into high-dimensional geometric space"
    assert "Embeddings represent" in q.explanation


def test_letter_answer_mapping():
    """Verify that when LLM returns 'A' or 'Option B', it resolves to the actual option text."""
    raw_json = [
        {
            "id": 1,
            "question": "Which index is used for vector search in StudyRAG?",
            "options": ["B-Tree", "FAISS IndexFlatIP", "Hash Map", "LSM Tree"],
            "correct_answer": "B",  # Option index 1 (second option)
            "explanation": "FAISS provides fast vector search.",
        },
        {
            "id": 2,
            "question": "What is PyMuPDF used for?",
            "options": ["PDF Text Extraction", "Vector Storage", "Audio Processing", "CSS Styling"],
            "correct_answer": "Option A",  # Option index 0
            "explanation": "PyMuPDF extracts text.",
        },
    ]

    questions = QuizGenerator._validate_and_sanitize_quiz(raw_json, expected_count=2)
    assert len(questions) == 2
    assert questions[0].correct_answer == "FAISS IndexFlatIP"
    assert questions[1].correct_answer == "PDF Text Extraction"


def test_malformed_json_rejection():
    """Verify that invalid items (less than 4 options, unresolvable answers) are discarded."""
    raw_json = [
        {
            # Invalid: only 3 options
            "id": 1,
            "question": "Too few options?",
            "options": ["Opt 1", "Opt 2", "Opt 3"],
            "correct_answer": "Opt 1",
            "explanation": "Invalid.",
        },
        {
            # Invalid: correct answer does not match any option
            "id": 2,
            "question": "Mismatch answer?",
            "options": ["Alpha", "Beta", "Gamma", "Delta"],
            "correct_answer": "Epsilon",
            "explanation": "Invalid.",
        },
        {
            # Valid question
            "id": 3,
            "question": "What does RAG stand for?",
            "options": [
                "Retrieval-Augmented Generation",
                "Random Access Gate",
                "Recursive Auto Graph",
                "Relative Action Group",
            ],
            "correct_answer": "Retrieval-Augmented Generation",
            "explanation": "Standard definition.",
        },
    ]

    questions = QuizGenerator._validate_and_sanitize_quiz(raw_json, expected_count=3)
    assert len(questions) == 1
    assert questions[0].question == "What does RAG stand for?"


def test_duplicate_question_deduplication():
    """Verify duplicate questions are filtered out."""
    raw_json = [
        {
            "question": "What is chunking?",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "explanation": "Exp",
        },
        {
            "question": "what is chunking?",  # Same question in different case
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "explanation": "Exp",
        },
    ]

    questions = QuizGenerator._validate_and_sanitize_quiz(raw_json, expected_count=2)
    assert len(questions) == 1


def test_quiz_grading_calculation():
    """Verify that grade_quiz accurately scores user answers and assigns letter grades."""
    questions = [
        QuizQuestion(1, "Q1", ["A", "B", "C", "D"], "A", "Exp 1"),
        QuizQuestion(2, "Q2", ["A", "B", "C", "D"], "B", "Exp 2"),
        QuizQuestion(3, "Q3", ["A", "B", "C", "D"], "C", "Exp 3"),
        QuizQuestion(4, "Q4", ["A", "B", "C", "D"], "D", "Exp 4"),
    ]

    # Case 1: 3 out of 4 correct (75%) -> Grade B
    user_answers = {1: "A", 2: "B", 3: "C", 4: "Wrong"}
    result = QuizGenerator.grade_quiz(questions, user_answers)

    assert result.total_questions == 4
    assert result.correct_answers == 3
    assert result.score_percentage == 75.0
    assert "B" in result.grade
    assert len(result.detailed_results) == 4
    assert result.detailed_results[0]["is_correct"] is True
    assert result.detailed_results[3]["is_correct"] is False

    # Case 2: 4 out of 4 (100%) -> Grade A+
    perfect_answers = {1: "A", 2: "B", 3: "C", 4: "D"}
    perfect_result = QuizGenerator.grade_quiz(questions, perfect_answers)
    assert perfect_result.score_percentage == 100.0
    assert "A+" in perfect_result.grade


def test_offline_extractive_quiz_generation():
    """Verify that QuizGenerator produces valid MCQs when no Groq API key is configured."""
    from unittest.mock import MagicMock
    from src.llm_client import LLMClient
    from src.vector_store import VectorStore
    from src.pdf_processor import ProcessedDocument, ProcessedPage
    from src.text_chunker import TextChunk

    # Mock unconfigured LLM
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.is_configured = False

    store = VectorStore()
    doc = ProcessedDocument("d_off", "os.pdf", 500, 1)
    chunks = [
        TextChunk("c1", "d_off", "os.pdf", 1, "- Virtual Memory: Provides address space isolation between processes.", 0, 80),
        TextChunk("c2", "d_off", "os.pdf", 1, "- Paging: Divides virtual memory into fixed size blocks called pages.", 1, 75),
        TextChunk("c3", "d_off", "os.pdf", 1, "- Deadlock: A condition where two or more processes are blocked waiting for each other.", 2, 90),
        TextChunk("c4", "d_off", "os.pdf", 1, "- Semaphore: A synchronization variable used to control access to shared resources.", 3, 90),
    ]
    store.add_document(doc, chunks)

    gen = QuizGenerator(store, mock_llm)
    questions = gen.generate_quiz(num_questions=3, difficulty="Medium")

    assert len(questions) == 3
    for q in questions:
        assert len(q.options) == 4
        assert q.correct_answer in q.options
        assert "Offline Mode" in q.difficulty
        assert q.source == "os.pdf, Page 1"
