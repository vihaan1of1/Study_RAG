"""StudyRAG Revision Tools Module.

Generates exam revision points, formula sheets, short-answer flashcard questions,
and last-minute exam cheat sheets strictly grounded in uploaded course materials.
"""

from __future__ import annotations

from typing import List, Optional

from src.llm_client import LLMClient
from src.vector_store import VectorStore


class RevisionToolsGenerator:
    """Specialized exam preparation and quick revision tools."""

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

    def generate_exam_revision_points(
        self, doc_ids: Optional[List[str]] = None, model: Optional[str] = None
    ) -> str:
        """Generate high-yield bulleted points likely to appear in examinations."""
        corpus = self._get_corpus(doc_ids)
        if not corpus:
            return "No document content available. Please upload and index documents first."

        if not self.llm_client.is_configured:
            target_chunks = [c for c in self.vector_store.chunks if not doc_ids or c.doc_id in doc_ids]
            points = []
            for c in target_chunks:
                for line in c.text.split("\n"):
                    line = line.strip()
                    if line.startswith(("-", "•", "*", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.")):
                        points.append(f"- **{line}** *(Page {c.page_number})*")
            if not points:
                points = [f"- **Core Concept**: {c.text[:180]}... *(Page {c.page_number})*" for c in target_chunks[:5]]

            return (
                "> ℹ️ *Extracted in Local Offline Mode. (Configure a free Groq API key in the sidebar for full AI exam coaching).*\n\n"
                "## 🎯 High-Yield Exam Revision Points\n\n"
                + "\n".join(points[:10])
            )

        prompt = (
            "You are an academic examiner preparing high-yield exam revision points for university students.\n"
            "Based strictly on the following study materials, generate 10 to 15 high-yield revision bullet points.\n\n"
            f"SOURCE MATERIALS:\n{corpus}\n\n"
            "Guidelines:\n"
            "- Group points under major topic headings.\n"
            "- Emphasize commonly tested principles, distinctions, edge cases, and architectural trade-offs.\n"
            "- Bold critical terms.\n"
            "- Include page citations where applicable."
        )

        messages = [
            {"role": "system", "content": "You are a university exam coach. Provide concise, test-relevant bullet points."},
            {"role": "user", "content": prompt},
        ]
        return self.llm_client.generate_chat_completion(
            messages=messages, model=model, temperature=0.2, max_tokens=1800
        )

    def extract_formulas_and_equations(
        self, doc_ids: Optional[List[str]] = None, model: Optional[str] = None
    ) -> str:
        """Extract mathematical, logical, or algorithmic formulas from text.

        If no formulas exist, clearly states that no formulas were found.
        """
        corpus = self._get_corpus(doc_ids)
        if not corpus:
            return "No document content available. Please upload and index documents first."

        if not self.llm_client.is_configured:
            target_chunks = [c for c in self.vector_store.chunks if not doc_ids or c.doc_id in doc_ids]
            found = []
            for c in target_chunks:
                for line in c.text.split("\n"):
                    line = line.strip()
                    if any(sym in line for sym in ("=", "CosineSimilarity", "MRR", "Sum_", "||", "O(", "Theta(", "Omega(")):
                        found.append(f"- **{line}** *(From {c.filename}, Page {c.page_number})*")

            if not found:
                return (
                    "> ℹ️ *Extracted in Local Offline Mode.*\n\n"
                    "No mathematical formulas or equations were found in the selected documents."
                )

            return (
                "> ℹ️ *Extracted in Local Offline Mode. (Configure a free Groq API key in the sidebar for LaTeX formatted breakdown).*\n\n"
                "## 📐 Mathematical Formulas & Equations Extracted\n\n"
                + "\n".join(found[:6])
            )

        prompt = (
            "Analyze the following study materials and extract any mathematical formulas, "
            "algorithmic complexities (Big-O), equations, or formal mathematical notations.\n\n"
            f"SOURCE MATERIALS:\n{corpus}\n\n"
            "CRITICAL RULE: If the source materials contain NO mathematical formulas or equations, "
            "explicitly state: 'No mathematical formulas or equations were found in the selected documents.' "
            "Do NOT invent formulas that are not present in the text.\n\n"
            "If formulas are present, for each formula provide:\n"
            "1. **Formula Name / Purpose**\n"
            "2. **Equation** (in LaTeX format like $...$ or $$...$$)\n"
            "3. **Variable Definitions** (what each symbol denotes)\n"
            "4. **Source Context** (page number)"
        )

        messages = [
            {
                "role": "system",
                "content": "Extract only actual formulas present in text. Never invent formulas.",
            },
            {"role": "user", "content": prompt},
        ]
        return self.llm_client.generate_chat_completion(
            messages=messages, model=model, temperature=0.1, max_tokens=1500
        )

    def generate_short_answer_questions(
        self, doc_ids: Optional[List[str]] = None, model: Optional[str] = None
    ) -> str:
        """Generate short-answer revision questions paired with model answers for self-study."""
        corpus = self._get_corpus(doc_ids)
        if not corpus:
            return "No document content available. Please upload and index documents first."

        if not self.llm_client.is_configured:
            target_chunks = [c for c in self.vector_store.chunks if not doc_ids or c.doc_id in doc_ids]
            qa_pairs = []
            for idx, c in enumerate(target_chunks[:5], start=1):
                sentences = [s.strip() for s in c.text.split(".") if len(s.strip()) > 35]
                if sentences:
                    first = sentences[0]
                    qa_pairs.append(
                        f"### Question {idx}: What does the course material establish regarding: '{first[:45]}...'?\n"
                        f"- **Model Answer**: {first}.\n"
                        f"- **Reference**: {c.filename} (Page {c.page_number})\n"
                    )
            return (
                "> ℹ️ *Generated in Local Offline Mode. (Configure a free Groq API key in the sidebar for full examination questions).*\n\n"
                "## ❓ Short-Answer Self-Study Questions\n\n"
                + "\n".join(qa_pairs)
            )

        prompt = (
            "Generate 6 to 10 high-quality short-answer revision questions based on the following text.\n"
            "Each question should test deep conceptual understanding rather than trivial trivia.\n\n"
            f"SOURCE MATERIALS:\n{corpus}\n\n"
            "Format each question as:\n"
            "### Question X: [Question text]\n"
            "- **Model Answer**: [Clear, precise 2-4 sentence answer grounded in the text]\n"
            "- **Key Marking Criteria**: [Key points required for full marks]\n"
            "- **Reference**: [Source document and page]\n"
        )

        messages = [
            {"role": "system", "content": "Generate rigorous academic self-study questions and answer keys."},
            {"role": "user", "content": prompt},
        ]
        return self.llm_client.generate_chat_completion(
            messages=messages, model=model, temperature=0.2, max_tokens=2200
        )

    def generate_last_minute_sheet(
        self, doc_ids: Optional[List[str]] = None, model: Optional[str] = None
    ) -> str:
        """Generate a dense, high-retention 1-page revision sheet for quick pre-exam review."""
        corpus = self._get_corpus(doc_ids)
        if not corpus:
            return "No document content available. Please upload and index documents first."

        if not self.llm_client.is_configured:
            target_chunks = [c for c in self.vector_store.chunks if not doc_ids or c.doc_id in doc_ids]
            snippets = [f"- {c.text[:120]}... *(Page {c.page_number})*" for c in target_chunks[:4]]

            return (
                "> ℹ️ *Compiled in Local Offline Mode. (Configure a free Groq API key in the sidebar for complete cheat-sheet synthesis).*\n\n"
                "## ⚡ 1-Page Last-Minute Exam Cheat Sheet\n\n"
                "### 1. ⚡ Top Must-Know Concepts\n"
                + "\n".join(snippets)
                + "\n\n### 2. ⚖️ Core Distinctions\n"
                "- Review primary trade-offs identified across lecture pages.\n\n"
                "### 3. 🎯 Fast Review Reminders\n"
                "- Ground all exam answers in definitions and formulas covered in course notes."
            )

        prompt = (
            "Create a condensed 'Last-Minute Exam Cheat Sheet' from the following course material.\n"
            "The goal is rapid, 10-minute pre-exam retention.\n\n"
            f"SOURCE MATERIALS:\n{corpus}\n\n"
            "Structure the sheet into 4 distinct quadrants:\n"
            "1. ⚡ **Top 5 Must-Know Concepts** (one sentence each)\n"
            "2. ⚖️ **Critical Distinctions & Comparisons** (e.g. Concept A vs Concept B)\n"
            "3. ⚠️ **Common Exam Pitfalls & Traps** (what students often confuse)\n"
            "4. 🎯 **Quick Reference Keywords & Mnemonic Summary**\n"
        )

        messages = [
            {"role": "system", "content": "You are creating a hyper-concise last-minute exam cheat sheet."},
            {"role": "user", "content": prompt},
        ]
        return self.llm_client.generate_chat_completion(
            messages=messages, model=model, temperature=0.2, max_tokens=2000
        )
