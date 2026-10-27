# StudyRAG — Professor Demonstration & Viva Examination Guide

This guide provides a step-by-step walkthrough to present **StudyRAG** to your professor and external examiners during project evaluations, lab vivas, and presentations.

---

## 1. Quick Demonstration Script (5-Minute Walkthrough)

### Step 1: Open the Application & Show the Dashboard (1 minute)
1. Start the application:
   ```powershell
   streamlit run app.py
   ```
2. Navigate to `http://localhost:8501`.
3. Highlight the modern academic design, logo, and active session status metrics:
   - Point out **Documents Indexed**, **Vector Chunks**, and **Quizzes Completed**.
   - Show the **Groq API Status** in the sidebar settings expander.

### Step 2: Upload Lecture Notes in the Document Library (1 minute)
1. In the sidebar, click on **📁 Document Library**.
2. Click **Browse files** and upload the included pre-generated sample document:
   `sample_docs/lecture_notes_rag.pdf`
   *(Or upload your actual course lecture notes).*
3. Click **📥 Process & Index 1 Document(s)**.
4. Point out the real-time processing indicator:
   - PyMuPDF text extraction.
   - Text chunking (800 character window with 150 character overlap).
   - In-memory vector embedding using local `all-MiniLM-L6-v2`.
5. Show the active document list displaying **3 Pages** and **Chunks Indexed**.
6. Open the **🔍 Inspect Extracted Text & Chunks** dropdown to show the professor that actual text was extracted with exact page numbers.

### Step 3: Demonstrate Conversational RAG with Citations (1.5 minutes)
1. In the sidebar, navigate to **💬 Chat with Documents**.
2. Type Question 1:
   > *"What is the difference between dense retrieval and sparse retrieval?"*
3. **Show the grounded response**: The model explains that sparse search uses exact keyword tokens like BM25, while dense retrieval maps queries and documents into continuous 384-dimensional latent semantic space.
4. **Show the Citation**:
   - Click to expand **📚 Supporting Sources**.
   - Point out: `[Source 1] lecture_notes_rag.pdf (Page 2)`.
   - Point out the exact snippet and similarity score.
5. Ask a follow-up question:
   > *"What mathematical formula is used to measure their similarity?"*
6. **Show conversational grounding**:
   - The model references the previous context and retrieves the Cosine Similarity formula from Page 2:
     $$\text{CosineSimilarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$$
   - Explain to the professor how multi-turn conversation works while staying strictly grounded in Page 2 of the notes.
7. Ask an out-of-scope question to prove hallucination prevention:
   > *"Who won the 2022 FIFA World Cup?"*
8. **Show the Grounding Guard**:
   - The model responds: *"Based on the provided documents, there is insufficient information to answer this question."*
   - Emphasize to the professor that StudyRAG refuses to hallucinate external facts not found in the course material.

### Step 4: Demonstrate Study Notes & Revision Tools (1 minute)
1. Click **📖 Study Notes**.
2. Click **💡 Key Concepts Breakdown** or **📖 Definitions Glossary**.
3. Point out how it extracts parametric vs non-parametric memory, grounding, and chunking into an academic table.
4. Click **📥 Download Notes as Markdown (.md)** to show that students can export structured notes for offline study.
5. Click **⚡ Revision Tools** and show the **⚡ Last-Minute Cheat Sheet** quadrant view.

### Step 5: Demonstrate Quiz Generation & Automatic Grading (1 minute)
1. Click **📝 Quiz Generator**.
2. Select:
   - Questions: **5**
   - Difficulty: **Medium**
3. Click **🚀 Generate New Quiz**.
4. Show the questions generated directly from the notes.
5. Select options for each question (intentionally get one wrong to demonstrate grading).
6. Click **📝 Submit Quiz & View Score**.
7. Show the calculated percentage, letter grade ($A, B, C, \dots$), and the color-coded feedback cards with full explanations and source references.
8. Point out how the **Quizzes Completed** metric increments on the dashboard.

---

## 2. Likely Viva Questions & Model Answers

### Q1: What is the difference between fine-tuning an LLM and using RAG?
> **Answer:** "Fine-tuning updates the internal weights (parametric memory) of the neural network on a specific dataset. It is expensive, slow to re-train when notes change, and still prone to hallucinations. RAG keeps the foundation model frozen and retrieves relevant factual passages dynamically from an external vector index (non-parametric memory). RAG is faster, requires no GPU training, allows instantaneous document updates, and provides verifiable page citations."

### Q2: Why did you choose FAISS instead of a cloud database like Pinecone or Weaviate?
> **Answer:** "For a student study assistant, privacy, speed, and zero cloud hosting costs are critical. FAISS runs locally on the CPU in-memory with C++ optimization. Because our document collections contain thousands of chunks rather than billions of vectors, `IndexFlatIP` performs exact cosine similarity search in milliseconds without subscription fees or network latency."

### Q3: Why do we use chunk overlap (e.g. 150 characters)?
> **Answer:** "If we cut text strictly at chunk boundaries without overlap, an important definition or formula could be split in half across two chunks. Overlap creates a sliding window so that any phrase spanning a boundary is captured completely in at least one chunk, preventing fragmented semantic context."

### Q4: How does StudyRAG prevent the LLM from inventing page numbers?
> **Answer:** "We never rely on the LLM to guess page numbers. During ingestion, PyMuPDF tags each chunk with its exact 1-indexed page number. When FAISS retrieves chunks, StudyRAG extracts the true stored metadata directly from memory to construct the citation cards displayed in the interface."

### Q5: What embedding model are you using, and what is its output dimension?
> **Answer:** "We use `sentence-transformers/all-MiniLM-L6-v2`. It outputs a 384-dimensional dense floating-point vector. It strikes an optimal trade-off between semantic accuracy, memory footprint, and CPU inference speed."

### Q6: What happens if an uploaded PDF is scanned or image-based?
> **Answer:** "PyMuPDF checks the character count of extracted text. If the total extracted text across the entire PDF is under 50 characters, StudyRAG detects it as a scanned/image-based PDF, halts ingestion, and alerts the student that OCR is required rather than falsely reporting success."

---

## 3. Technical Highlights to Emphasize

1. **Modular Architecture**: Separate single-responsibility modules (`pdf_processor.py`, `text_chunker.py`, `vector_store.py`, `rag_engine.py`, `quiz_generator.py`).
2. **Robust Error Handling**: JSON validation and repair loops for LLM output, graceful handling of missing API keys, corrupted files, and rate limits.
3. **Automated Test Suite**: 19 automated tests with `pytest` verifying extraction, chunking, retrieval, scoring, and end-to-end integration.
4. **State Persistence**: Streamlit session state keeps documents, vectors, chat history, and quiz state preserved across user interactions without re-reading files on every click.
