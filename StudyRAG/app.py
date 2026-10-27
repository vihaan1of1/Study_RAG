"""StudyRAG — An Intelligent RAG-Based Study Assistant.

Main Streamlit Application.
Provides interactive Chat, Study Notes, Revision Tools, Quiz Generation, and Document Management.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional
import streamlit as st

import config
from src.pdf_processor import PDFProcessor, ProcessedDocument
from src.text_chunker import TextChunker
from src.vector_store import VectorStore
from src.llm_client import LLMClient
from src.rag_engine import RAGEngine
from src.study_notes import StudyNotesGenerator
from src.revision_tools import RevisionToolsGenerator
from src.quiz_generator import QuizGenerator, QuizQuestion, QuizResult
from src.utils import build_markdown_export

# Set page config
st.set_page_config(
    page_title=f"{config.APP_TITLE} — AI Study Assistant",
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_custom_css() -> None:
    """Inject custom academic theme CSS."""
    css_path = config.ASSETS_DIR / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialize persistent session variables."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore(config.DEFAULT_EMBEDDING_MODEL)

    if "chunker" not in st.session_state:
        st.session_state.chunker = TextChunker(
            chunk_size=config.DEFAULT_CHUNK_SIZE,
            chunk_overlap=config.DEFAULT_CHUNK_OVERLAP,
        )

    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs: Dict[str, ProcessedDocument] = {}

    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Dict] = []

    if "quiz_attempts" not in st.session_state:
        st.session_state.quiz_attempts = 0

    if "current_quiz" not in st.session_state:
        st.session_state.current_quiz: Optional[List[QuizQuestion]] = None

    if "quiz_user_answers" not in st.session_state:
        st.session_state.quiz_user_answers: Dict[int, str] = {}

    if "quiz_results" not in st.session_state:
        st.session_state.quiz_results: Optional[QuizResult] = None

    if "study_notes_output" not in st.session_state:
        st.session_state.study_notes_output: str = ""

    if "revision_output" not in st.session_state:
        st.session_state.revision_output: str = ""

    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = config.DEFAULT_LLM_PROVIDER

    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = config.get_gemini_api_key()

    if "gemini_model" not in st.session_state:
        st.session_state.gemini_model = config.DEFAULT_GEMINI_MODEL

    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = config.get_groq_api_key()

    if "groq_model" not in st.session_state:
        st.session_state.groq_model = config.DEFAULT_GROQ_MODEL

    if "current_page" not in st.session_state:
        st.session_state.current_page = "📊 Dashboard"


def get_llm_client() -> LLMClient:
    """Instantiate LLM client with active session configuration."""
    provider = st.session_state.get("llm_provider", config.DEFAULT_LLM_PROVIDER)
    if provider == "gemini":
        return LLMClient(
            provider="gemini",
            api_key=st.session_state.gemini_api_key,
            default_model=st.session_state.gemini_model,
        )
    elif provider == "groq":
        return LLMClient(
            provider="groq",
            api_key=st.session_state.groq_api_key,
            default_model=st.session_state.groq_model,
        )
    return LLMClient(provider="offline")


def get_active_model() -> str:
    """Return the name of the currently active model."""
    provider = st.session_state.get("llm_provider", config.DEFAULT_LLM_PROVIDER)
    if provider == "gemini":
        return st.session_state.get("gemini_model", config.DEFAULT_GEMINI_MODEL)
    if provider == "groq":
        return st.session_state.get("groq_model", config.DEFAULT_GROQ_MODEL)
    return "local-extractive"


def is_active_llm_ready() -> bool:
    """Check if the selected provider is ready with an API key."""
    provider = st.session_state.get("llm_provider", config.DEFAULT_LLM_PROVIDER)
    if provider == "gemini":
        return bool(st.session_state.get("gemini_api_key"))
    if provider == "groq":
        return bool(st.session_state.get("groq_api_key"))
    return True


# --- Section Handlers ---


def render_sidebar() -> str:
    """Render sidebar navigation, document summary metrics, and API settings."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                <span style="font-size: 2rem;">{config.APP_ICON}</span>
                <div>
                    <h2 style="margin:0; font-size:1.4rem; font-weight:800; color:#1e293b;">{config.APP_TITLE}</h2>
                    <p style="margin:0; font-size:0.78rem; color:#64748b;">Study Companion</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        nav_options = [
            "📊 Dashboard",
            "💬 Chat with Documents",
            "📖 Study Notes",
            "⚡ Revision Tools",
            "📝 Quiz Generator",
            "📁 Document Library",
        ]

        # Sync nav selection with session state
        default_idx = 0
        if st.session_state.current_page in nav_options:
            default_idx = nav_options.index(st.session_state.current_page)

        selected_page = st.radio(
            "Navigation",
            options=nav_options,
            index=default_idx,
            label_visibility="collapsed",
        )
        st.session_state.current_page = selected_page

        st.markdown("---")

        # Session Metrics
        total_docs = st.session_state.vector_store.total_documents
        total_chunks = st.session_state.vector_store.total_chunks
        quiz_attempts = st.session_state.quiz_attempts

        st.markdown("### 📈 Session Status")
        col1, col2 = st.columns(2)
        col1.metric("Docs", total_docs)
        col2.metric("Chunks", total_chunks)
        st.metric("Quizzes Completed", quiz_attempts)

        st.markdown("---")

        # AI Provider & Model Configuration Expander
        with st.expander("⚙️ AI Provider & Settings", expanded=not is_active_llm_ready()):
            provider_labels = [
                "Google Gemini (Recommended)",
                "Groq (Llama 3)",
                "Local Offline Mode",
            ]
            current_p = st.session_state.get("llm_provider", "gemini")
            def_p_idx = 0 if current_p == "gemini" else (1 if current_p == "groq" else 2)

            selected_label = st.selectbox(
                "LLM Provider",
                options=provider_labels,
                index=def_p_idx,
            )
            new_p = "gemini" if "Gemini" in selected_label else ("groq" if "Groq" in selected_label else "offline")
            if new_p != st.session_state.llm_provider:
                st.session_state.llm_provider = new_p
                st.rerun()

            if new_p == "gemini":
                gemini_key_input = st.text_input(
                    "Gemini API Key",
                    type="password",
                    value=st.session_state.gemini_api_key,
                    help="Google Gemini API key",
                )
                if gemini_key_input != st.session_state.gemini_api_key:
                    st.session_state.gemini_api_key = gemini_key_input
                    st.rerun()

                gem_model = st.selectbox(
                    "Gemini Model",
                    options=config.SUPPORTED_GEMINI_MODELS,
                    index=0 if st.session_state.gemini_model not in config.SUPPORTED_GEMINI_MODELS else config.SUPPORTED_GEMINI_MODELS.index(st.session_state.gemini_model),
                )
                st.session_state.gemini_model = gem_model

                if st.session_state.gemini_api_key:
                    st.success(f"✅ Gemini Active ({st.session_state.gemini_model})")
                else:
                    st.warning("⚠️ Enter your Gemini key to enable Gemini 3.8 Flash.")

            elif new_p == "groq":
                groq_key_input = st.text_input(
                    "Groq API Key",
                    type="password",
                    value=st.session_state.groq_api_key,
                    help="Obtain a free API key at console.groq.com",
                )
                if groq_key_input != st.session_state.groq_api_key:
                    st.session_state.groq_api_key = groq_key_input
                    st.rerun()

                groq_model = st.selectbox(
                    "Groq Model",
                    options=config.SUPPORTED_GROQ_MODELS,
                    index=0 if st.session_state.groq_model not in config.SUPPORTED_GROQ_MODELS else config.SUPPORTED_GROQ_MODELS.index(st.session_state.groq_model),
                )
                st.session_state.groq_model = groq_model

                if st.session_state.groq_api_key:
                    st.success(f"✅ Groq Active ({st.session_state.groq_model})")
                else:
                    st.warning("⚠️ Enter your Groq key to enable Llama 3.")

            else:
                st.info("💡 Local Offline Mode: Generates answers, notes, and quizzes using local extractive RAG algorithms without external API calls.")

            st.caption(f"Embeddings: `{config.DEFAULT_EMBEDDING_MODEL.split('/')[-1]}` (Local)")

    return selected_page


def render_dashboard() -> None:
    """Render main welcome dashboard with metrics, quick actions, and guidance."""
    st.markdown(
        f"""
        <div class="brand-container">
            <span class="brand-badge">{config.APP_ICON} {config.APP_TITLE}</span>
            <div>
                <h1 class="brand-title">Welcome to StudyRAG</h1>
                <p class="brand-tagline">{config.APP_TAGLINE}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_docs = st.session_state.vector_store.total_documents
    total_chunks = st.session_state.vector_store.total_chunks
    quiz_attempts = st.session_state.quiz_attempts

    # Metric Cards Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total_docs}</div>
                <div class="metric-label">Documents Indexed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total_chunks}</div>
                <div class="metric-label">Vector Chunks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{quiz_attempts}</div>
                <div class="metric-label">Quizzes Completed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        is_ready = is_active_llm_ready()
        provider_name = st.session_state.llm_provider.capitalize()
        status_text = f"{provider_name} Ready" if is_ready else "Key Needed"
        status_color = "#16a34a" if is_ready else "#ca8a04"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{status_color}; font-size:1.4rem;">{status_text}</div>
                <div class="metric-label">LLM Engine Status</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    # Quick Actions
    st.markdown("### 🚀 Quick Actions")
    q1, q2, q3, q4 = st.columns(4)

    with q1:
        if st.button("📁 Upload PDFs", use_container_width=True):
            st.session_state.current_page = "📁 Document Library"
            st.rerun()

    with q2:
        if st.button("💬 Ask Questions", use_container_width=True, disabled=total_docs == 0):
            st.session_state.current_page = "💬 Chat with Documents"
            st.rerun()

    with q3:
        if st.button("📖 Generate Notes", use_container_width=True, disabled=total_docs == 0):
            st.session_state.current_page = "📖 Study Notes"
            st.rerun()

    with q4:
        if st.button("📝 Start Quiz", use_container_width=True, disabled=total_docs == 0):
            st.session_state.current_page = "📝 Quiz Generator"
            st.rerun()

    st.write("")

    # Guide Banner & Feature Highlights
    st.markdown(
        """
        <div class="guide-banner">
            <h3 style="margin:0 0 8px 0; color:#1e1b4b; font-size:1.15rem;">🎓 How to get started with StudyRAG</h3>
            <ol style="margin:0; padding-left:20px; color:#312e81; font-size:0.95rem; line-height:1.6;">
                <li><b>Upload Lecture Notes or Papers:</b> Head to <i>Document Library</i> and upload one or more academic PDFs. Text is extracted, chunked, and embedded into local FAISS vector space.</li>
                <li><b>Ask Context-Grounded Questions:</b> In <i>Chat with Documents</i>, ask any question. StudyRAG retrieves relevant passages with page-level citations.</li>
                <li><b>Create Study Notes & Cheat Sheets:</b> Generate summaries, definitions, and last-minute exam revision points in <i>Study Notes</i> and <i>Revision Tools</i>.</li>
                <li><b>Test Your Knowledge:</b> Generate 5, 10, or 15 multiple-choice questions in <i>Quiz Generator</i> with explanations and automatic grading.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">🔍 Grounded RAG with Page Citations</div>
                <div class="feature-desc">Every answer cites the source PDF filename and exact original page number. Inspect the supporting text snippets directly to verify statements.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">⚡ Local Privacy & High Speed</div>
                <div class="feature-desc">Embeddings are computed locally using Sentence Transformers. Vector similarity search runs in memory using FAISS CPU with zero database hosting fees.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">🎯 Multiple-Choice Exam Simulator</div>
                <div class="feature-desc">Self-assess with syllabus-grounded quizzes. Select difficulty, submit answers, and receive detailed explanations and source references.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">📥 Instant Study Material Export</div>
                <div class="feature-desc">Download generated chapter summaries, key definition glossaries, and last-minute cheat sheets directly as structured Markdown files.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_document_library() -> None:
    """Render Document Library for PDF upload, validation, inspection, and removal."""
    st.markdown("## 📁 Document Library & Ingestion")
    st.markdown("Upload academic PDFs, inspect extracted pages, and manage indexed knowledge chunks.")

    # Upload Section
    uploaded_files = st.file_uploader(
        "Upload Academic PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload lecture slides, textbooks, or research papers (machine-readable PDF format).",
    )

    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if not any(d.filename == f.name for d in st.session_state.processed_docs.values())
        ]

        if new_files:
            if st.button(f"📥 Process & Index {len(new_files)} Document(s)", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, file in enumerate(new_files):
                    status_text.text(f"Extracting text from '{file.name}'...")
                    file_bytes = file.read()

                    # Step 1: Process PDF
                    doc = PDFProcessor.process_pdf(file_bytes=file_bytes, filename=file.name)

                    if not doc.is_valid:
                        st.error(f"Failed to process '{file.name}': {doc.error}")
                        continue

                    # Step 2: Chunk Document
                    status_text.text(f"Chunking '{file.name}' (800 chars / 150 overlap)...")
                    chunks = st.session_state.chunker.chunk_document(doc)

                    # Step 3: Embed & Store in FAISS
                    status_text.text(f"Generating embeddings for '{file.name}'...")
                    num_added = st.session_state.vector_store.add_document(doc, chunks)

                    st.session_state.processed_docs[doc.doc_id] = doc
                    st.success(
                        f"✅ Successfully indexed **{file.name}**: {doc.total_pages} page(s), "
                        f"{num_added} chunks."
                    )
                    progress_bar.progress((idx + 1) / len(new_files))

                status_text.text("Processing complete!")
                st.rerun()

    st.write("")
    st.markdown("### 📚 Active Document Collection")

    docs = list(st.session_state.processed_docs.values())
    if not docs:
        st.info("No documents uploaded yet. Upload a PDF above to begin.")
        return

    # Document Management Table/Cards
    for doc in docs:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
            col1.markdown(f"**📄 {doc.filename}**")
            col2.caption(f"{doc.total_pages} Pages")
            doc_chunks = [c for c in st.session_state.vector_store.chunks if c.doc_id == doc.doc_id]
            col3.caption(f"{len(doc_chunks)} Chunks")
            col4.caption(f"{doc.file_size_bytes / 1024:.1f} KB")

            if col5.button("🗑️ Remove", key=f"del_{doc.doc_id}"):
                st.session_state.vector_store.remove_document(doc.doc_id)
                del st.session_state.processed_docs[doc.doc_id]
                st.success(f"Removed '{doc.filename}' and rebuilt vector index.")
                st.rerun()

    # Search & Inspection Expander
    with st.expander("🔍 Inspect Extracted Text & Chunks"):
        inspect_doc = st.selectbox(
            "Select Document to Inspect",
            options=docs,
            format_func=lambda d: d.filename,
        )
        if inspect_doc:
            st.markdown(f"**Document ID:** `{inspect_doc.doc_id}` | **Pages:** `{inspect_doc.total_pages}`")
            page_num = st.number_input("Page Number", min_value=1, max_value=max(1, inspect_doc.total_pages), value=1)
            target_page = next((p for p in inspect_doc.pages if p.page_number == page_num), None)
            if target_page:
                st.text_area(f"Page {page_num} Text ({target_page.char_count} chars)", value=target_page.text, height=200)
            else:
                st.write("Page not found.")


def render_chat_with_documents() -> None:
    """Render RAG Question-Answering conversational interface with source citations."""
    st.markdown("## 💬 Chat with Documents")
    st.markdown("Ask conceptual, factual, or analytical questions grounded strictly in your uploaded materials.")

    docs = list(st.session_state.processed_docs.values())
    if not docs:
        st.warning("⚠️ No documents available. Please upload PDFs in the **Document Library** first.")
        if st.button("Go to Document Library"):
            st.session_state.current_page = "📁 Document Library"
            st.rerun()
        return

    # Top Filters Bar
    f_col1, f_col2, f_col3 = st.columns([2, 1, 1])

    with f_col1:
        doc_options = {d.doc_id: d.filename for d in docs}
        selected_doc_ids = st.multiselect(
            "Filter by Documents (default: all documents)",
            options=list(doc_options.keys()),
            format_func=lambda did: doc_options[did],
            default=[],
            help="Select one or more documents to narrow retrieval scope, or leave empty to search all.",
        )
        effective_filter = selected_doc_ids if selected_doc_ids else None

    with f_col2:
        top_k = st.slider("Top-K Passages", min_value=2, max_value=8, value=config.DEFAULT_TOP_K)

    with f_col3:
        st.write("")
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")

    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # If message contains verified citations, render source cards
            citations = message.get("citations", [])
            if citations:
                with st.expander(f"📚 Supporting Sources ({len(citations)} citations)"):
                    for c in citations:
                        st.markdown(
                            f"""
                            <div class="citation-box">
                                <span class="citation-pill">[Source {c['source_index']}] {c['filename']}</span>
                                <span class="similarity-badge">Page {c['page_number']} | Sim: {c.get('similarity_score', 0.0):.2f}</span>
                                <p style="margin: 6px 0 0 0; color: #334155; font-size:0.88rem;">"{c['snippet']}"</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Chat Input Box
    user_query = st.chat_input("Ask a question about your uploaded documents...")

    if user_query:
        # Append user query to UI history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Execute RAG Pipeline
        with st.chat_message("assistant"):
            active_model = get_active_model()
            spinner_msg = f"Retrieving passages and formulating answer with {st.session_state.llm_provider.capitalize()}..." if is_active_llm_ready() else "Retrieving top passage via FAISS vector search..."
            with st.spinner(spinner_msg):
                llm = get_llm_client()
                engine = RAGEngine(
                    vector_store=st.session_state.vector_store,
                    llm_client=llm,
                )

                response = engine.answer_question(
                    query=user_query,
                    chat_history=st.session_state.chat_history[:-1],
                    top_k=top_k,
                    doc_filter=effective_filter,
                    model=active_model,
                )

                st.markdown(response.answer)

                citations_data = [
                    {
                        "source_index": c.source_index,
                        "filename": c.filename,
                        "page_number": c.page_number,
                        "snippet": c.snippet,
                        "similarity_score": c.similarity_score,
                    }
                    for c in response.citations
                ]

                if response.citations:
                    with st.expander(f"📚 Supporting Sources ({len(response.citations)} citations)"):
                        for c in response.citations:
                            st.markdown(
                                f"""
                                <div class="citation-box">
                                    <span class="citation-pill">[Source {c.source_index}] {c.filename}</span>
                                    <span class="similarity-badge">Page {c.page_number} | Sim: {c.similarity_score:.2f}</span>
                                    <p style="margin: 6px 0 0 0; color: #334155; font-size:0.88rem;">"{c.snippet}"</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "citations": citations_data,
                    }
                )


def render_study_notes() -> None:
    """Render Study Notes & Summarization interface with downloadable output."""
    st.markdown("## 📖 Academic Study Notes & Summarization")
    st.markdown("Generate grounded chapter summaries, conceptual frameworks, and definition glossaries.")

    docs = list(st.session_state.processed_docs.values())
    if not docs:
        st.warning("⚠️ No documents available. Please upload PDFs in the **Document Library** first.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        doc_options = {d.doc_id: d.filename for d in docs}
        selected_doc_ids = st.multiselect(
            "Select Documents to Synthesize",
            options=list(doc_options.keys()),
            format_func=lambda did: doc_options[did],
            default=list(doc_options.keys()),
            help="Choose which documents to include in the note synthesis.",
        )

    with col2:
        st.write("")
        st.caption(f"Synthesizing from {len(selected_doc_ids)} selected document(s)")

    st.write("")

    # Action Buttons Row
    b1, b2, b3, b4 = st.columns(4)

    llm = get_llm_client()
    generator = StudyNotesGenerator(st.session_state.vector_store, llm)
    active_model = get_active_model()

    target_docs = selected_doc_ids if selected_doc_ids else None

    if b1.button("📋 Executive Summary", use_container_width=True):
        with st.spinner("Generating summary..."):
            output = generator.generate_short_summary(
                doc_ids=target_docs, model=active_model
            )
            st.session_state.study_notes_output = output

    if b2.button("📑 Detailed Chapter Notes", use_container_width=True):
        with st.spinner("Generating detailed study guide..."):
            output = generator.generate_detailed_summary(
                doc_ids=target_docs, model=active_model
            )
            st.session_state.study_notes_output = output

    if b3.button("💡 Key Concepts Breakdown", use_container_width=True):
        with st.spinner("Extracting critical concepts..."):
            output = generator.generate_key_concepts(
                doc_ids=target_docs, model=active_model
            )
            st.session_state.study_notes_output = output

    if b4.button("📖 Definitions Glossary", use_container_width=True):
        with st.spinner("Extracting terminology & definitions..."):
            output = generator.generate_definitions(
                doc_ids=target_docs, model=active_model
            )
            st.session_state.study_notes_output = output

    st.markdown("---")

    # Output Display & Download
    if st.session_state.study_notes_output:
        st.markdown(st.session_state.study_notes_output)

        md_download = build_markdown_export(
            title="StudyRAG Study Notes",
            content=st.session_state.study_notes_output,
            metadata={
                "Document Count": len(selected_doc_ids),
                "Model": active_model,
            },
        )

        st.download_button(
            label="📥 Download Notes as Markdown (.md)",
            data=md_download,
            file_name="StudyRAG_Notes.md",
            mime="text/markdown",
        )
    else:
        st.info("Select an action above to generate study notes based on your documents.")


def render_revision_tools() -> None:
    """Render Revision Tools: exam bullet points, formulas, and last-minute sheets."""
    st.markdown("## ⚡ Exam Revision Tools")
    st.markdown("Prepare for tests with high-yield revision points, formula extractions, and practice flashcards.")

    docs = list(st.session_state.processed_docs.values())
    if not docs:
        st.warning("⚠️ No documents available. Please upload PDFs in the **Document Library** first.")
        return

    doc_options = {d.doc_id: d.filename for d in docs}
    selected_doc_ids = st.multiselect(
        "Source Documents",
        options=list(doc_options.keys()),
        format_func=lambda did: doc_options[did],
        default=list(doc_options.keys()),
    )
    target_docs = selected_doc_ids if selected_doc_ids else None

    r1, r2, r3, r4 = st.columns(4)
    llm = get_llm_client()
    rev_tools = RevisionToolsGenerator(st.session_state.vector_store, llm)
    active_model = get_active_model()

    if r1.button("🎯 High-Yield Exam Points", use_container_width=True):
        with st.spinner("Extracting high-yield exam points..."):
            res = rev_tools.generate_exam_revision_points(target_docs, active_model)
            st.session_state.revision_output = res

    if r2.button("📐 Formulas & Equations", use_container_width=True):
        with st.spinner("Searching for formulas, Big-O, and math expressions..."):
            res = rev_tools.extract_formulas_and_equations(target_docs, active_model)
            st.session_state.revision_output = res

    if r3.button("❓ Short-Answer Questions", use_container_width=True):
        with st.spinner("Generating practice test questions..."):
            res = rev_tools.generate_short_answer_questions(target_docs, active_model)
            st.session_state.revision_output = res

    if r4.button("⚡ Last-Minute Cheat Sheet", use_container_width=True):
        with st.spinner("Compiling 1-page high-retention cheat sheet..."):
            res = rev_tools.generate_last_minute_sheet(target_docs, active_model)
            st.session_state.revision_output = res

    st.markdown("---")

    if st.session_state.revision_output:
        st.markdown(st.session_state.revision_output)

        md_download = build_markdown_export(
            title="StudyRAG Revision Sheet",
            content=st.session_state.revision_output,
            metadata={"Source Documents": len(selected_doc_ids)},
        )
        st.download_button(
            label="📥 Download Revision Sheet (.md)",
            data=md_download,
            file_name="StudyRAG_Revision_Sheet.md",
            mime="text/markdown",
        )
    else:
        st.info("Click one of the revision options above to begin.")


def render_quiz_generator() -> None:
    """Render Multiple-Choice Quiz generator, interactive exam interface, and scoring."""
    st.markdown("## 📝 Multiple-Choice Quiz Generator")
    st.markdown("Generate practice quizzes directly from your course materials to test your understanding.")

    docs = list(st.session_state.processed_docs.values())
    if not docs:
        st.warning("⚠️ No documents available. Please upload PDFs in the **Document Library** first.")
        return

    # Quiz Configuration Bar
    q_col1, q_col2, q_col3, q_col4 = st.columns([2, 1, 1, 1.2])

    with q_col1:
        doc_options = {d.doc_id: d.filename for d in docs}
        selected_doc_ids = st.multiselect(
            "Quiz Source Documents",
            options=list(doc_options.keys()),
            format_func=lambda did: doc_options[did],
            default=list(doc_options.keys()),
        )
        target_docs = selected_doc_ids if selected_doc_ids else None

    with q_col2:
        num_q = st.selectbox("Questions", options=[5, 10, 15], index=0)

    with q_col3:
        difficulty = st.selectbox("Difficulty", options=["Easy", "Medium", "Hard"], index=1)

    with q_col4:
        st.write("")
        gen_clicked = st.button("🚀 Generate New Quiz", type="primary", use_container_width=True)

    if gen_clicked:
        active_model = get_active_model()
        is_ready = is_active_llm_ready()
        provider_name = st.session_state.llm_provider.capitalize()
        spinner_msg = f"Generating {num_q} {difficulty} questions with {provider_name} AI..." if is_ready else f"Extracting {num_q} {difficulty} questions from course notes (Offline Mode)..."
        with st.spinner(spinner_msg):
            llm = get_llm_client()
            quiz_gen = QuizGenerator(st.session_state.vector_store, llm)
            try:
                questions = quiz_gen.generate_quiz(
                    num_questions=num_q,
                    difficulty=difficulty,
                    doc_ids=target_docs,
                    model=active_model,
                )
                st.session_state.current_quiz = questions
                st.session_state.quiz_user_answers = {}
                st.session_state.quiz_results = None
                st.rerun()
            except Exception as exc:
                st.error(f"Quiz generation error: {str(exc)}")

    st.markdown("---")

    # Render Active Quiz
    questions = st.session_state.current_quiz

    if not questions:
        st.info("Configure your quiz settings above and click **Generate New Quiz** to start.")
        return

    # Check if quiz has been submitted
    if st.session_state.quiz_results is not None:
        # Results Screen
        res = st.session_state.quiz_results
        st.markdown(
            f"""
            <div class="guide-banner" style="text-align:center;">
                <h2 style="margin:0; color:#1e1b4b;">Quiz Results: {res.score_percentage}%</h2>
                <p style="font-size:1.2rem; font-weight:700; color:#4f46e5; margin:6px 0;">Grade: {res.grade}</p>
                <p style="margin:0; color:#475569;">You answered <b>{res.correct_answers}</b> out of <b>{res.total_questions}</b> questions correctly.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        # Detailed Question Review
        for item in res.detailed_results:
            is_correct = item["is_correct"]
            card_class = "result-correct" if is_correct else "result-incorrect"
            badge = '<span class="badge-success">CORRECT</span>' if is_correct else '<span class="badge-danger">INCORRECT</span>'

            st.markdown(
                f"""
                <div class="{card_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <b>Question {item['id']}</b>
                        {badge}
                    </div>
                    <div style="font-size:1.05rem; font-weight:700; margin-bottom:10px;">{item['question']}</div>
                    <p style="margin:4px 0;"><b>Your Answer:</b> {item['user_answer']}</p>
                    <p style="margin:4px 0;"><b>Correct Answer:</b> {item['correct_answer']}</p>
                    <p style="margin:8px 0 0 0; color:#475569; font-size:0.9rem;"><i>Explanation:</i> {item['explanation']}</p>
                    <p style="margin:4px 0 0 0; color:#64748b; font-size:0.8rem;"><b>Source:</b> {item['source']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("🔄 Retake This Quiz", type="primary"):
            st.session_state.quiz_user_answers = {}
            st.session_state.quiz_results = None
            st.rerun()

    else:
        # Quiz Taking Mode
        st.markdown(f"### 📋 Practice Quiz ({len(questions)} Questions)")
        st.caption("Select the best option for each question. Answers will be scored after submission.")

        for q in questions:
            with st.container():
                st.markdown(
                    f"""
                    <div class="quiz-question-box">
                        <span class="quiz-q-num">Question {q.id} of {len(questions)}</span>
                        <div class="quiz-q-title">{q.question}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                current_choice = st.session_state.quiz_user_answers.get(q.id)
                choice_idx = None
                if current_choice and current_choice in q.options:
                    choice_idx = q.options.index(current_choice)

                selected = st.radio(
                    f"Select answer for Q{q.id}:",
                    options=q.options,
                    index=choice_idx,
                    key=f"q_{q.id}",
                    label_visibility="collapsed",
                )
                st.session_state.quiz_user_answers[q.id] = selected
                st.write("")

        # Submit Quiz Button
        st.write("")
        if st.button("📝 Submit Quiz & View Score", type="primary", use_container_width=True):
            # Check for unanswered questions
            answered_count = len([ans for ans in st.session_state.quiz_user_answers.values() if ans])
            if answered_count < len(questions):
                st.warning(f"Note: You have answered {answered_count} of {len(questions)} questions.")

            results = QuizGenerator.grade_quiz(questions, st.session_state.quiz_user_answers)
            st.session_state.quiz_results = results
            st.session_state.quiz_attempts += 1
            st.rerun()


# --- Main Application Execution ---


def main() -> None:
    """Application entrypoint."""
    load_custom_css()
    init_session_state()

    selected_page = render_sidebar()

    if selected_page == "📊 Dashboard":
        render_dashboard()
    elif selected_page == "💬 Chat with Documents":
        render_chat_with_documents()
    elif selected_page == "📖 Study Notes":
        render_study_notes()
    elif selected_page == "⚡ Revision Tools":
        render_revision_tools()
    elif selected_page == "📝 Quiz Generator":
        render_quiz_generator()
    elif selected_page == "📁 Document Library":
        render_document_library()


if __name__ == "__main__":
    main()
