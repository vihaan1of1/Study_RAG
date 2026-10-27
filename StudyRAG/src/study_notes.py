"""StudyRAG Study Notes and Summarization Module.

Generates concise summaries, detailed breakdowns, key concepts, and definitions
glossaries from uploaded academic documents with chunk batching and token limits.
"""

from __future__ import annotations

from typing import List, Optional

from src.llm_client import LLMClient
from src.vector_store import VectorStore


class StudyNotesGenerator:
    """Produces structured academic notes and summaries from document collections."""

    # Maximum characters sent to LLM per prompt to stay well within context windows
    MAX_CONTEXT_CHARS = 12000

    def __init__(self, vector_store: VectorStore, llm_client: LLMClient):
        self.vector_store = vector_store
        self.llm_client = llm_client

    def _get_context_corpus(self, doc_ids: Optional[List[str]] = None) -> str:
        """Assemble a representative text corpus from selected documents with provenance."""
        if not self.vector_store.chunks:
            return ""

        target_chunks = self.vector_store.chunks
        if doc_ids:
            target_chunks = [c for c in target_chunks if c.doc_id in doc_ids]

        if not target_chunks:
            return ""

        total_chars = sum(len(c.text) for c in target_chunks)

        # If fits within budget, include all
        if total_chars <= self.MAX_CONTEXT_CHARS:
            selected = target_chunks
        else:
            # Hierarchical / stratified sampling: pick evenly spaced chunks across the document
            step = max(1, len(target_chunks) * self.MAX_CONTEXT_CHARS // total_chars)
            selected = target_chunks[::max(1, len(target_chunks) // 15)]

        corpus_parts = []
        for c in selected:
            corpus_parts.append(f"[{c.filename}, Page {c.page_number}]:\n{c.text}")

        return "\n\n---\n\n".join(corpus_parts)

    def generate_short_summary(
        self, doc_ids: Optional[List[str]] = None, model: Optional[str] = None
    ) -> str:
        """Generate a concise executive summary of key ideas (2-3 paragraphs)."""
        corpus = self._get_context_corpus(doc_ids)
        if not corpus:
            return "No document content available. Please upload and index documents first."

        if not self.llm_client.is_configured:
            # Offline extractive summary
            target_chunks = [c for c in self.vector_store.chunks if not doc_ids or c.doc_id in doc_ids]
            first_chunks = target_chunks[:3]
            summary_points = []
            for c in first_chunks:
                lines = [l.strip() for l in c.text.split("\n") if len(l.strip()) > 30]
                if lines:
                    summary_points.append(f"- **From {c.filename} (Page {c.page_number})**: {lines[0]}")

            return (
                "> ℹ️ *Generated in Local Offline Mode. (Configure a free Groq API key in the sidebar for generative synthesis).*\n\n"
                "### 📋 Executive Summary\n"
                "**Core Overview & Objectives:**\n"
                "This study collection covers foundational principles, architectures, and quantitative methods extracted from your course materials.\n\n"
                "**Key Takeaways:**\n"
                + "\n".join(summary_points)
                + "\n\n**Conclusion:**\n"
                "The documents provide comprehensive coverage of these topics for exam preparation."
            )

        prompt = (
            "You are an academic study assistant. Provide a concise, high-impact summary of the "
            "following study materials. Outline the core objective, main arguments, and key conclusions.\n\n"
            f"SOURCE MATERIALS:\n{corpus}\n\n"
            "Format the summary with:\n"
            "- **Overview & Core Objective**\n"
            "- **Key Takeaways (3-5 bullet points)**\n"
            "- **Conclusion**\n\n"
            "Base your summary strictly on the provided text."
        )

        messages = [
            {
                "role": "system",
                "content": "You are a precise academic summarizer. Summarize only facts present in the text.",
            },
            {"role": "user", "content": prompt},
        ]
        return self.llm_client.generate_chat_completion(
            messages=messages, model=model, temperature=0.2, max_tokens=1200
        )

    def generate_detailed_summary(
        self, doc_ids: Optional[List[str]] = None, model: Optional[str] = None
    ) -> str:
        """Generate a comprehensive, chapter-by-chapter or topic-by-topic summary."""
        corpus = self._get_context_corpus(doc_ids)
        if not corpus:
            return "No document content available. Please upload and index documents first."

        if not self.llm_client.is_configured:
            target_chunks = [c for c in self.vector_store.chunks if not doc_ids or c.doc_id in doc_ids]
            sections = []
            for idx, c in enumerate(target_chunks[:8], start=1):
                clean_lines = [l.strip() for l in c.text.split("\n") if l.strip()]
                heading = clean_lines[0] if clean_lines else f"Section {idx}"
                body = " ".join(clean_lines[1:4]) if len(clean_lines) > 1 else c.text[:200]
                sections.append(f"### {idx}. {heading}\n*Source: {c.filename} (Page {c.page_number})*\n\n{body}\n")

            return (
                "> ℹ️ *Generated in Local Offline Mode. (Configure a free Groq API key in the sidebar for full synthesis).*\n\n"
                "## 📑 Detailed Study Guide & Chapter Breakdown\n\n"
                + "\n".join(sections)
            )

        prompt = (
            "You are an academic professor. Provide a detailed, comprehensive study guide "
            "and chapter-level breakdown based on the following materials:\n\n"
            f"SOURCE MATERIALS:\n{corpus}\n\n"
            "Include:\n"
            "1. **Introduction and Scope**\n"
            "2. **Detailed Section-by-Section Breakdown** with subheadings\n"
            "3. **Critical Mechanisms, Processes, or Methodologies**\n"
            "4. **Synthesis & Implications**\n\n"
            "Include inline page citations where available (e.g. [Page X]). Ground strictly in the text."
        )

        messages = [
            {
                "role": "system",
                "content": "You are a detailed academic writer. Create structured, thorough study guides.",
            },
            {"role": "user", "content": prompt},
        ]
        return self.llm_client.generate_chat_completion(
            messages=messages, model=model, temperature=0.2, max_tokens=2200
        )

    def extract_key_concepts(
        self, doc_ids: Optional[List[str]] = None, model: Optional[str] = None
    ) -> str:
        """Extract and explain the most critical conceptual frameworks and themes."""
        corpus = self._get_context_corpus(doc_ids)
        if not corpus:
            return "No document content available. Please upload and index documents first."

        if not self.llm_client.is_configured:
            import re
            target_chunks = [c for c in self.vector_store.chunks if not doc_ids or c.doc_id in doc_ids]
            concepts = []
            seen = set()
            for c in target_chunks:
                for line in c.text.split("\n"):
                    m = re.match(r"^[-*•]?\s*([A-Za-z0-9\s()_-]{3,35})\s*:\s*(.{25,250})", line.strip())
                    if m and m.group(1).lower() not in seen:
                        seen.add(m.group(1).lower())
                        concepts.append(
                            f"### 💡 {m.group(1).strip()}\n"
                            f"- **What it is**: {m.group(2).strip()}\n"
                            f"- **Source Reference**: {c.filename} (Page {c.page_number})\n"
                        )
            if not concepts:
                concepts = [f"### 💡 {c.filename} Key Concepts\n- {c.text[:200]}..." for c in target_chunks[:3]]

            return (
                "> ℹ️ *Extracted in Local Offline Mode. (Configure a free Groq API key in the sidebar for full conceptual synthesis).*\n\n"
                "## 💡 Critical Academic Concepts\n\n"
                + "\n".join(concepts[:8])
            )

        prompt = (
            "Extract the 6 to 10 most critical academic concepts, theories, or models from the "
            "following source text:\n\n"
            f"SOURCE MATERIALS:\n{corpus}\n\n"
            "For each concept, format as:\n"
            "### [Concept Name]\n"
            "- **What it is**: Clear explanation in simple academic terms.\n"
            "- **Why it matters**: Significance in the context of the subject.\n"
            "- **Example or Application**: Concrete example mentioned in the text.\n"
            "- **Source Reference**: Page or section where discussed.\n\n"
            "Do not invent concepts not in the text."
        )

        messages = [
            {
                "role": "system",
                "content": "Extract and explain academic concepts strictly based on source text.",
            },
            {"role": "user", "content": prompt},
        ]
        return self.llm_client.generate_chat_completion(
            messages=messages, model=model, temperature=0.2, max_tokens=1800
        )

    def extract_definitions(
        self, doc_ids: Optional[List[str]] = None, model: Optional[str] = None
    ) -> str:
        """Extract a structured glossary of technical terms and their precise definitions."""
        corpus = self._get_context_corpus(doc_ids)
        if not corpus:
            return "No document content available. Please upload and index documents first."

        if not self.llm_client.is_configured:
            import re
            target_chunks = [c for c in self.vector_store.chunks if not doc_ids or c.doc_id in doc_ids]
            table_rows = []
            seen = set()
            for c in target_chunks:
                for line in c.text.split("\n"):
                    m = re.match(r"^[-*•]?\s*([A-Za-z0-9\s()_-]{3,35})\s*:\s*(.{25,250})", line.strip())
                    if m and m.group(1).lower() not in seen:
                        seen.add(m.group(1).lower())
                        table_rows.append(f"| **{m.group(1).strip()}** | {m.group(2).strip()} | {c.filename} (Page {c.page_number}) |")

            if not table_rows:
                table_rows.append("| **Course Material** | Extracted from uploaded lecture documents | All Pages |")

            return (
                "> ℹ️ *Extracted in Local Offline Mode. (Configure a free Groq API key in the sidebar for full glossary generation).*\n\n"
                "## 📖 Glossary of Terms & Definitions\n\n"
                "| Term | Academic Definition | Source Location |\n"
                "| :--- | :--- | :--- |\n"
                + "\n".join(table_rows[:12])
            )

        prompt = (
            "Extract an alphabetical Glossary of Terms and Definitions from the text below.\n\n"
            f"SOURCE MATERIALS:\n{corpus}\n\n"
            "Format as a Markdown table or clean definition list:\n"
            "| Term | Academic Definition | Context / Notes |\n"
            "| :--- | :--- | :--- |\n\n"
            "Extract at least 8 to 15 terms if present. Only include terms explicitly discussed in the text."
        )

        messages = [
            {
                "role": "system",
                "content": "Extract accurate glossaries and definitions directly from study materials.",
            },
            {"role": "user", "content": prompt},
        ]
        return self.llm_client.generate_chat_completion(
            messages=messages, model=model, temperature=0.2, max_tokens=1600
        )
