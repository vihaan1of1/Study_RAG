"""StudyRAG Quiz Generator Module.

Generates rigorous, syllabus-aligned multiple-choice questions (MCQs) in structured JSON,
validates options and ground-truth answers, and provides scoring and explanations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.llm_client import LLMClient
from src.vector_store import VectorStore


@dataclass
class QuizQuestion:
    """Validated single Multiple-Choice Question."""
    id: int
    question: str
    options: List[str]  # Exactly 4 options
    correct_answer: str  # Must match one item in options
    explanation: str
    source: str = "Course Material"
    difficulty: str = "Medium"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "options": self.options,
            "correct_answer": self.correct_answer,
            "explanation": self.explanation,
            "source": self.source,
            "difficulty": self.difficulty,
        }


@dataclass
class QuizResult:
    """Result summary of a submitted quiz attempt."""
    total_questions: int
    correct_answers: int
    score_percentage: float
    grade: str
    detailed_results: List[Dict[str, Any]]


class QuizGenerator:
    """Engine for creating and scoring curriculum-grounded quizzes."""

    MAX_CONTEXT_CHARS = 12000

    def __init__(self, vector_store: VectorStore, llm_client: LLMClient):
        self.vector_store = vector_store
        self.llm_client = llm_client

    def _get_corpus(self, doc_ids: Optional[List[str]] = None) -> str:
        """Fetch representative context text from vector store."""
        if not self.vector_store.chunks:
            return ""

        chunks = self.vector_store.chunks
        if doc_ids:
            chunks = [c for c in chunks if c.doc_id in doc_ids]

        if not chunks:
            return ""

        total_chars = sum(len(c.text) for c in chunks)
        if total_chars <= self.MAX_CONTEXT_CHARS:
            selected = chunks
        else:
            selected = chunks[::max(1, len(chunks) // 15)]

        parts = [f"[{c.filename}, Page {c.page_number}]:\n{c.text}" for c in selected]
        return "\n\n---\n\n".join(parts)

    def generate_quiz(
        self,
        num_questions: int = 5,
        difficulty: str = "Medium",
        doc_ids: Optional[List[str]] = None,
        model: Optional[str] = None,
    ) -> List[QuizQuestion]:
        """Generate a validated list of MCQs based on document content.

        Falls back to intelligent local extractive quiz generation if no Groq API key is configured.
        """
        corpus = self._get_corpus(doc_ids)
        if not corpus:
            raise ValueError("No indexed documents found. Please upload documents first.")

        # Local Offline Fallback if Groq API key is not configured
        if not self.llm_client.is_configured:
            return self._generate_extractive_quiz(
                num_questions=num_questions,
                difficulty=difficulty,
                doc_ids=doc_ids,
            )

        prompt = (
            f"Generate an academic multiple-choice quiz with exactly {num_questions} questions "
            f"at '{difficulty}' difficulty based STRICTLY on the provided source materials.\n\n"
            f"SOURCE MATERIALS:\n{corpus}\n\n"
            "REQUIREMENTS FOR EACH QUESTION:\n"
            "1. Must test concepts, facts, or applications explicitly found in the materials.\n"
            "2. Provide EXACTLY 4 distinct, plausible options.\n"
            "3. Exactly ONE option must be the correct answer.\n"
            "4. Provide a clear academic explanation referencing why the answer is correct.\n"
            "5. Cite the source document/page where possible.\n\n"
            "OUTPUT FORMAT:\n"
            "Return a JSON array of objects with the exact schema:\n"
            "[\n"
            "  {\n"
            '    "id": 1,\n'
            '    "question": "What is ...?",\n'
            '    "options": ["Option A", "Option B", "Option C", "Option D"],\n'
            '    "correct_answer": "Option A",\n'
            '    "explanation": "Because...",\n'
            '    "source": "Document.pdf, Page X"\n'
            "  }\n"
            "]\n\n"
            "Ensure 'correct_answer' is identical to one of the 4 strings in 'options'. "
            "Output ONLY valid JSON."
        )

        system_prompt = (
            "You are a rigorous university examination coordinator. You generate flawless, "
            "strictly verified multiple-choice test questions in JSON format."
        )

        raw_json = self.llm_client.generate_structured_json(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            max_retries=2,
        )

        validated_questions = self._validate_and_sanitize_quiz(
            raw_json, expected_count=num_questions, difficulty=difficulty
        )

        if not validated_questions:
            raise ValueError(
                "Failed to parse and validate quiz questions from LLM response. "
                "Please retry generation."
            )

        return validated_questions

    def _generate_extractive_quiz(
        self,
        num_questions: int = 5,
        difficulty: str = "Medium",
        doc_ids: Optional[List[str]] = None,
    ) -> List[QuizQuestion]:
        """Generate curriculum-grounded MCQs using local document extraction when offline."""
        import random

        chunks = self.vector_store.chunks
        if doc_ids:
            chunks = [c for c in chunks if c.doc_id in doc_ids]

        candidates = []
        for c in chunks:
            lines = c.text.split("\n")
            for line in lines:
                line = line.strip()
                m = re.match(r"^[-*•]?\s*([A-Za-z0-9\s()_-]{3,40})\s*:\s*(.{25,250})", line)
                if m:
                    candidates.append((m.group(1).strip(), m.group(2).strip(), c.filename, c.page_number))
                else:
                    m2 = re.match(
                        r"^([A-Z][A-Za-z0-9\s()_-]{2,35})\s+(is|refers to|measures|represents|calculates)\s+(.{25,250})",
                        line,
                    )
                    if m2:
                        candidates.append((m2.group(1).strip(), f"{m2.group(2).strip()} {m2.group(3).strip()}", c.filename, c.page_number))

        # Fallback if few definitions found: extract declarative sentences
        if len(candidates) < 4:
            for c in chunks:
                sentences = re.split(r"(?<=[.!?])\s+", c.text)
                for s in sentences:
                    s = s.strip()
                    if 35 <= len(s) <= 220:
                        words = s.split()
                        key_concept = " ".join(words[:3])
                        candidates.append((key_concept, s, c.filename, c.page_number))

        seen_terms = set()
        unique_candidates = []
        for term, defn, fname, pnum in candidates:
            clean_term = term.strip().strip("-*• ")
            if clean_term.lower() in ("course notes", "module 4", "chapter 1", "introduction", "section"):
                continue
            if clean_term.lower() not in seen_terms and len(defn) > 20:
                seen_terms.add(clean_term.lower())
                unique_candidates.append((clean_term, defn.strip(), fname, pnum))

        if len(unique_candidates) < 4:
            raise ValueError(
                "The selected documents contain insufficient text to construct multiple-choice questions. "
                "Please upload a more comprehensive course document."
            )

        selected_candidates = unique_candidates[:num_questions]
        all_defs = [c[1] for c in unique_candidates]

        questions = []
        for idx, (term, correct_def, fname, pnum) in enumerate(selected_candidates, start=1):
            distractors = [d for d in all_defs if d != correct_def]
            # Ensure at least 3 distinct distractors
            extra_distractors = [
                f"An external optimization parameter unrelated to {term}.",
                f"A legacy heuristic replaced by {term}.",
                f"A non-convergent mathematical approximation of {term}.",
            ]
            for extra in extra_distractors:
                if len(distractors) < 3:
                    distractors.append(extra)

            options = [correct_def] + distractors[:3]
            rng = random.Random(idx * 73)
            rng.shuffle(options)

            q_text = f"According to the course notes, which statement accurately describes '{term}'?"
            if "formula" in term.lower() or "similarity" in term.lower():
                q_text = f"What is the definition or role of '{term}' in this subject?"

            questions.append(
                QuizQuestion(
                    id=idx,
                    question=q_text,
                    options=options,
                    correct_answer=correct_def,
                    explanation=f"Referenced from {fname} (Page {pnum}): '{term}: {correct_def}'",
                    source=f"{fname}, Page {pnum}",
                    difficulty=f"{difficulty} (Offline Mode)",
                )
            )

        return questions

    @classmethod
    def _validate_and_sanitize_quiz(
        cls,
        data: Any,
        expected_count: int,
        difficulty: str = "Medium",
    ) -> List[QuizQuestion]:
        """Validate and fix raw JSON quiz structure."""
        if isinstance(data, dict):
            # If wrapped in a dictionary like {"questions": [...]} or {"quiz": [...]}
            for key in ["questions", "quiz", "data", "items"]:
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break

        if not isinstance(data, list):
            return []

        validated: List[QuizQuestion] = []
        seen_questions = set()

        letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}

        for idx, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue

            q_text = str(item.get("question", "")).strip()
            if not q_text or q_text.lower() in seen_questions:
                continue

            raw_options = item.get("options")
            if not isinstance(raw_options, list):
                continue

            # Clean and deduplicate options
            options = [str(opt).strip() for opt in raw_options if str(opt).strip()]
            if len(options) != 4 or len(set(options)) != 4:
                # Need strictly 4 distinct options
                continue

            correct = str(item.get("correct_answer", "")).strip()
            # Handle if model returned letter "A", "B", "C", "D" or "Option A"
            resolved_correct = None
            if correct in options:
                resolved_correct = correct
            elif correct.upper() in letter_map:
                resolved_correct = options[letter_map[correct.upper()]]
            elif re.match(r"^Option\s*([A-D])$", correct, re.IGNORECASE):
                match = re.match(r"^Option\s*([A-D])$", correct, re.IGNORECASE)
                letter = match.group(1).upper()
                resolved_correct = options[letter_map[letter]]
            else:
                # Check for partial or fuzzy match
                for opt in options:
                    if correct.lower() == opt.lower() or opt.lower() in correct.lower():
                        resolved_correct = opt
                        break

            if not resolved_correct:
                continue

            explanation = str(
                item.get("explanation", f"The correct answer is {resolved_correct}.")
            ).strip()
            source = str(item.get("source", "Course Materials")).strip()

            seen_questions.add(q_text.lower())
            validated.append(
                QuizQuestion(
                    id=idx,
                    question=q_text,
                    options=options,
                    correct_answer=resolved_correct,
                    explanation=explanation,
                    source=source,
                    difficulty=difficulty,
                )
            )

        return validated

    @staticmethod
    def grade_quiz(
        questions: List[QuizQuestion], user_answers: Dict[int, str]
    ) -> QuizResult:
        """Calculate score, percentage, and detailed answer-by-answer review."""
        if not questions:
            return QuizResult(
                total_questions=0,
                correct_answers=0,
                score_percentage=0.0,
                grade="N/A",
                detailed_results=[],
            )

        total = len(questions)
        correct_count = 0
        details = []

        for q in questions:
            user_choice = user_answers.get(q.id)
            is_correct = bool(user_choice and user_choice == q.correct_answer)
            if is_correct:
                correct_count += 1

            details.append(
                {
                    "id": q.id,
                    "question": q.question,
                    "options": q.options,
                    "user_answer": user_choice or "Unanswered",
                    "correct_answer": q.correct_answer,
                    "is_correct": is_correct,
                    "explanation": q.explanation,
                    "source": q.source,
                }
            )

        percentage = round((correct_count / total) * 100, 1)

        if percentage >= 90:
            grade = "A+ (Outstanding)"
        elif percentage >= 80:
            grade = "A (Excellent)"
        elif percentage >= 70:
            grade = "B (Good)"
        elif percentage >= 60:
            grade = "C (Satisfactory)"
        elif percentage >= 50:
            grade = "D (Pass)"
        else:
            grade = "F (Needs Revision)"

        return QuizResult(
            total_questions=total,
            correct_answers=correct_count,
            score_percentage=percentage,
            grade=grade,
            detailed_results=details,
        )
