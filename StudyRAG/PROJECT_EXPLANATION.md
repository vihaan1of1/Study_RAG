# StudyRAG — In-Depth Technical & Academic Explanation

This document is prepared specifically for computer engineering students, professors, and viva examiners to explain every layer of the **StudyRAG** system from foundational principles to runtime mechanics.

---

## 1. What is RAG and Why is it Essential?

### The Core Problem with Standard LLMs
Pre-trained Large Language Models (LLMs) like GPT-4 or Llama 3 possess vast general linguistic capability. However, in an academic or domain-specific setting, relying solely on an LLM creates critical failure modes:
1. **Parametric Memory Limitations**: The model's knowledge is frozen at the end of its training run (the training cutoff date).
2. **Hallucination**: When an LLM does not know a specific fact from a proprietary university lecture slide or recent research paper, it generates statistically probable text that sounds authoritative but is factually false.
3. **Black-Box Answers**: An ordinary LLM cannot tell you: *"I found this equation on Page 2, Paragraph 3 of your lecture notes."*

### What is RAG?
**Retrieval-Augmented Generation (RAG)** is an AI architecture that decouples *linguistic reasoning* from *knowledge storage*. Instead of forcing the LLM to memorize all facts:
- **Non-Parametric Storage**: Course documents are stored in an external vector index.
- **Dynamic Retrieval**: When a student asks a question, the system queries the vector index for the top-$k$ most relevant passages.
- **Context Injection**: The retrieved excerpts are inserted into the LLM's prompt as the sole factual source.
- **Grounded Generation**: The LLM is instructed: *"Answer the student's question using ONLY the provided passages, and cite the document name and page number for every assertion."*

---

## 2. Step-by-Step Data Flow: From PDF Upload to Grounded Answer

```
[Student Uploads PDF]
        │
        ▼
[PyMuPDF (fitz) Text Extraction] ──► Extracts text per page; records (filename, page_number)
        │
        ▼
[Text Sanitization] ───────────────► Cleans null bytes, excessive whitespace, control characters
        │
        ▼
[Text Chunking] ───────────────────► Sliding window (800 chars, 150 overlap); assigns Chunk IDs
        │
        ▼
[Embedding Generation] ────────────► all-MiniLM-L6-v2 maps text chunks into 384-dim dense vectors
        │
        ▼
[FAISS Vector Store] ──────────────► Normalized vectors indexed in FAISS (IndexFlatIP)
        │
============================ RUNTIME QUESTION ANSWERING ============================
        │
[Student Asks Question] ───────────► "What is the formula for cosine similarity?"
        │
        ▼
[Query Vectorization] ─────────────► Question mapped into identical 384-dim semantic space
        │
        ▼
[FAISS Cosine Similarity Search] ──► Inner-product search retrieves top-5 closest vectors
        │
        ▼
[Provenance & Citation Prep] ──────► Chunks mapped back to metadata (Page 2, snippet, score)
        │
        ▼
[Context Assembly & Grounding] ────► System prompt forbids speculation; injects [Source X] tags
        │
        ▼
[Groq LLM Generation] ─────────────► Model reads context and crafts grounded explanation
        │
        ▼
[UI Presentation] ─────────────────► Displays answer with interactive citation cards & snippets
```

---

## 3. How PDF Text Extraction Works

StudyRAG uses **PyMuPDF (`pymupdf` / `fitz`)**, a high-performance C-backed PDF parsing engine.
1. **Binary Signature Verification**: StudyRAG verifies that the binary byte stream starts with `%PDF-` before parsing.
2. **Page Iteration**: The document is read page by page in a loop (`for page in doc:`).
3. **Metadata Association**: The extracted text from each page is bound to its 1-indexed page number (`page_num = page_idx + 1`).
4. **Scanned PDF Detection**: If the total characters extracted across all pages is under 50, the document is flagged as a scanned image or bitmap. StudyRAG warns the student that optical character recognition (OCR) is required rather than giving a false positive of successful indexing.

---

## 4. What Chunking and Overlap Mean

### Why Can't We Index an Entire Document as One String?
- **Embedding Dilution**: If you embed a 50-page document into a single 384-dimensional vector, the embedding becomes a generalized blur of everything and loses the specific detail of individual equations or theorems.
- **Context Window Limits**: While modern LLMs have larger context windows, filling prompts with 100 pages of irrelevant text increases inference costs, latency, and "lost-in-the-middle" attention degradation.

### The Sliding Window Strategy
StudyRAG splits text into segments of approximately **800 characters**:
- **Why 800 characters?** Approximately 120-160 words—long enough to contain a coherent thought or definition, yet focused enough to retain high semantic specificity.
- **Why 150 characters of overlap?** When a critical concept spans two adjacent chunks, a hard cut might split the definition in half (e.g., placing the concept name in Chunk 1 and its explanation in Chunk 2). Overlap guarantees that the full relational meaning appears intact in at least one chunk.
- **Natural Boundary Preservation**: The chunker scans backwards from the window boundary to split at paragraph breaks (`\n\n`) or sentence ends (`. `, `? `) rather than cutting words in half.

---

## 5. What are Embeddings?

An **embedding** is a numerical representation of semantic meaning mapped into a continuous high-dimensional geometric space.

StudyRAG uses `sentence-transformers/all-MiniLM-L6-v2`:
- It is a 6-layer MiniLM model fine-tuned on over 1 billion sentence pairs using contrastive loss.
- It transforms any text string into a vector of **384 floating-point numbers** ($\mathbb{R}^{384}$).
- **Semantic Proximity**: Sentences with similar meanings (e.g., *"What is paging?"* and *"Explain virtual memory page allocation"*) produce vectors that point in nearly identical directions in 384-dimensional space, even though they use different words.

---

## 6. How FAISS Similarity Search Works

**FAISS (Facebook AI Similarity Search)** is an industry-standard C++ library developed by Meta AI for efficient similarity search over dense vectors.

### Cosine Similarity and Normalized Vectors
Cosine similarity between query vector $\mathbf{q}$ and chunk vector $\mathbf{d}$ is defined as:
$$\text{CosineSimilarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$$

StudyRAG normalizes all embeddings using $L_2$-normalization ($\|\mathbf{x}\|_2 = 1.0$) upon generation. When vectors have unit length ($\|\mathbf{q}\| = 1$ and $\|\mathbf{d}\| = 1$), cosine similarity simplifies directly to the dot product (inner product):
$$\text{CosineSimilarity}(\mathbf{q}, \mathbf{d}) = \mathbf{q} \cdot \mathbf{d}$$

StudyRAG uses `faiss.IndexFlatIP` (Exact Inner Product Flat Index):
- It performs an exhaustive search across all stored vectors.
- Because the dataset consists of course documents (hundreds to thousands of chunks rather than billions of vectors), exhaustive search runs in milliseconds with **zero quantization error or approximation loss**.

---

## 7. How Verifiable Citations are Produced

In inferior RAG implementations, developers ask the LLM to "invent citations" or type out page numbers in its response. This leads to hallucinated page citations.

StudyRAG enforces **provenance-grounded citations**:
1. When FAISS returns the top-$k$ nearest chunks, StudyRAG retrieves their actual stored metadata: `filename`, `page_number`, `chunk_id`, and `similarity_score`.
2. Each passage is tagged as `[Source 1]`, `[Source 2]`, etc.
3. In the UI, StudyRAG constructs expandable source cards containing the exact page number and text snippet stored in memory. The student can click open each source to visually confirm that the answer corresponds verbatim to that page of their lecture notes.

---

## 8. How Quiz Generation Works

The Quiz Generator does not generate arbitrary trivia from the general web:
1. **Context Sampling**: It extracts a representative sample of text chunks from the student's selected course documents.
2. **Strict Few-Shot Prompting**: It instructs Groq Llama 3 to output a valid JSON array of multiple-choice questions testing concepts, formulas, and distinctions explicitly present in the text.
3. **Python Schema Validation**: StudyRAG does not blindly display whatever JSON the LLM returns. It executes automated verification:
   - Does each question have exactly 4 distinct options?
   - Does `correct_answer` strictly match one of those 4 options?
   - If the model returned `"A"` or `"Option B"`, StudyRAG automatically resolves the letter to the exact option string.
   - Are there any duplicate questions?
4. **Stateful Scoring**: Answers are stored in Streamlit session state and revealed only upon submission, preventing cheating and providing detailed explanations.

---

## 9. Handling Missing Information & Edge Cases

When a student asks a question whose answer does not exist in the uploaded documents:
- **Grounding Directive**: The system prompt explicitly commands: *"If the context does not contain sufficient information to answer the question, state explicitly: 'Based on the provided documents, there is insufficient information to answer this question.' Do NOT fabricate facts."*
- **Empty Retrieval Guard**: If vector similarity scores fall below relevant thresholds or no documents are indexed, StudyRAG immediately notifies the user rather than sending an empty query to the LLM.
- **Document Removal Isolation**: When a document is deleted in the Document Library, StudyRAG completely wipes and rebuilds the active FAISS index from the remaining documents. This ensures zero "zombie chunks" leak into future queries.

---

## 10. Technical Limitations & Future Improvements

1. **Scanned Documents (OCR)**: Currently, StudyRAG extracts digital text from PDF streams. Image-based or hand-written notes require adding an OCR pre-processing step (e.g. `tesseract` or `easyocr`).
2. **Hybrid Search (BM25 + Dense Vectors)**: While dense embeddings excel at semantic concepts, sparse keyword search (BM25) is superior for rare serial numbers, variable names, or specific acronyms. Combining both with Reciprocal Rank Fusion (RRF) represents the next frontier for StudyRAG.
3. **Cross-Encoder Re-ranking**: Adding a second-stage cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score the top-20 retrieved passages before sending the top-5 to the LLM would further improve retrieval precision.
