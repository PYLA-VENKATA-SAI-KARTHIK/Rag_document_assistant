# Interview Preparation Guide for RAG Document Assistant

This guide is designed for a final-year BTech candidate who built this project and wants to defend it confidently in an interview.

It focuses on:

- Common interview questions.
- Questions that actually test real understanding.
- Deep-dive questions an expert interviewer would ask.
- What a strong answer should contain.
- What the candidate still needs to learn to fully own this project.

## 1. Project Summary

The project is a Retrieval-Augmented Generation (RAG) system for PDF documents.

High-level flow:

1. Upload a PDF.
2. Extract and clean the text.
3. Split the text into chunks.
4. Generate embeddings for each chunk.
5. Store the embeddings in FAISS.
6. Embed the user query.
7. Retrieve the most relevant chunks.
8. Send retrieved context to Groq LLM.
9. Return a grounded answer.

Core files:

- `app.py` - Streamlit UI and session flow.
- `src/pdf_loader.py` - PDF parsing and chunking.
- `src/embeddings.py` - Embedding generation.
- `src/vector_store.py` - FAISS indexing and search.
- `src/llm_handler.py` - Prompt building and Groq call.
- `src/rag_pipeline.py` - End-to-end CLI pipeline.

## 2. What an Interviewer Usually Wants to Know

An interviewer is typically checking these things:

- Do you understand the problem statement?
- Can you explain the architecture clearly?
- Do you know why each component exists?
- Can you defend design choices?
- Do you understand tradeoffs and limitations?
- Can you diagnose failure cases?
- Can you explain how to improve it further?

If the candidate only says "I used FAISS, embeddings, and Groq," that is not enough.
They should be able to explain the pipeline, why it works, and where it can fail.

## 3. Common Interview Questions

These are the questions that are asked most often.

### Project and Motivation

1. What problem does this project solve?
2. Why did you choose a RAG architecture instead of training a model?
3. What is the main value of your project compared to a simple document search tool?
4. Why did you build a PDF-based assistant?
5. What makes your project different from a normal chatbot?

### Architecture

6. Explain the full system architecture.
7. Walk me through the data flow from upload to answer.
8. Why did you choose Streamlit for the frontend?
9. Why did you choose FAISS for retrieval?
10. Why did you choose SentenceTransformers for embeddings?
11. Why did you choose Groq for generation?

### Document Processing

12. How do you extract text from PDF files?
13. Why do you clean the extracted text?
14. Why do you chunk the text?
15. Why did you choose the chunk size and overlap values?
16. What happens if the PDF text extraction is poor?

### Embeddings and Retrieval

17. What is an embedding?
18. Why are embeddings useful in RAG?
19. What is the dimension of the embedding model you used?
20. Why did you use vector search instead of keyword search?
21. How does FAISS similarity search work?
22. What does top-k retrieval mean?

### LLM and Prompting

23. Why do you send retrieved chunks to the LLM?
24. How do you make sure the model answers only from context?
25. What is prompt engineering in this project?
26. What happens if the answer is not in the document?
27. Why do you use temperature 0.3?

### Performance and Limitations

28. What are the limitations of your current system?
29. How would you handle very large PDFs?
30. How would you handle long documents with multiple sections?
31. What if the relevant answer is split across two chunks?
32. What if the PDF contains tables or scanned images?

### Security and Deployment

33. How do you store API keys securely?
34. Why should `.env` never be committed?
35. How would you deploy this project safely?
36. What would you do if your API key leaked?

## 4. Questions That Actually Test Real Knowledge

These are better questions than the usual surface-level ones. If a candidate can answer these well, they understand the project deeply.

### Core Understanding

1. Why is RAG better than giving the full document to the LLM?
2. Why can embeddings capture semantic similarity better than keyword matching?
3. What is the difference between chunking for retrieval and chunking for summarization?
4. Why is chunk overlap important?
5. Why can too-large chunks hurt retrieval quality?
6. Why can too-small chunks hurt answer quality?
7. Why is retrieval a bottleneck in RAG systems?

### Retrieval Quality

8. What are the signs that retrieval is failing?
9. How would you debug a wrong answer: retrieval problem or generation problem?
10. How would you evaluate whether the retrieved chunks are actually relevant?
11. What metrics would you use for retrieval quality?
12. What metrics would you use for answer quality?

### System Behavior

13. Why can the same question produce different answers sometimes?
14. What happens if the query embedding is poor?
15. What happens if the context window is too small?
16. What happens if the prompt includes irrelevant chunks?
17. Why is hallucination still possible in RAG?

### Engineering Tradeoffs

18. Why did you choose a local vector store instead of a database-backed solution?
19. Why did you use a flat FAISS index instead of an approximate one?
20. What tradeoff exists between speed and accuracy in vector search?
21. Why store chunks separately instead of only storing the embeddings?
22. Why is session state used in the Streamlit app?

## 5. Deep-Dive Questions an Expert Interviewer May Ask

These are the questions that separate shallow understanding from real ownership.

### Retrieval and Chunking

1. How did you choose `chunk_size=500` and `chunk_overlap=50`? What would happen if you doubled both values?
2. Why does overlapping text help retrieval recall?
3. How would you design chunking differently for legal documents, resumes, and research papers?
4. What is the impact of chunk boundaries on answer quality?
5. How would you combine parent-child chunking or hierarchical retrieval in this project?

### Embeddings

6. What does the `all-MiniLM-L6-v2` model encode, and what are its practical strengths and weaknesses?
7. Why is embedding normalization important in some retrieval setups?
8. How would you replace the current embedding model with a domain-specific model?
9. How would multilingual documents change your embedding choice?
10. How would you cache embeddings to avoid recomputation?

### FAISS and Similarity Search

11. Why is `IndexFlatL2` simple but not always scalable?
12. When would you use HNSW or IVF instead of flat search?
13. What is the difference between L2 distance and cosine similarity?
14. How would you convert this project to cosine-based retrieval?
15. How would you persist and reload the FAISS index?

### Prompting and LLM Control

16. How does your prompt reduce hallucination?
17. What specific prompt changes would improve factual grounding?
18. Why did you choose a low temperature?
19. How would you stop the model from over-explaining?
20. What would a better answer format look like for this app?

### Production Readiness

21. How would you add authentication to this app?
22. How would you log usage safely without leaking sensitive document content?
23. How would you rate-limit API calls?
24. How would you handle large file uploads in production?
25. How would you monitor quality regressions after deployment?

### Robustness

26. What should happen if Groq is unavailable?
27. What should happen if the PDF is corrupted?
28. What should happen if the user uploads a scanned image PDF?
29. What should happen if no relevant chunks are found?
30. What should happen if the extracted text is mostly noise?

## 6. Expert Answers Guide

Below are the kinds of answers that sound strong in an interview.

### Question: What problem does this project solve?

Strong answer:

"This project solves document question answering. Instead of manually searching through a PDF, the user can upload a document and ask natural-language questions. The system retrieves relevant text chunks using semantic search and then uses an LLM to produce a grounded answer."

What the interviewer wants to hear:

- You understand the user pain point.
- You understand retrieval plus generation.
- You can explain the value clearly.

### Question: Why did you choose RAG?

Strong answer:

"I chose RAG because it is cheaper, faster, and more accurate for document-specific knowledge than training a custom model. The document can be indexed once and queried many times. It also allows the answer to stay grounded in the source text."

### Question: Why chunk the document?

Strong answer:

"The embedding model and LLM both work better on smaller, focused text segments. Chunking helps preserve local context, improves retrieval precision, and avoids sending an entire document into the model. I used overlap so important context at the boundaries is not lost."

### Question: Why use embeddings?

Strong answer:

"Embeddings convert text into dense vectors where semantically similar content is closer in vector space. This lets the system find relevant passages even when the question and answer use different words."

### Question: Why FAISS?

Strong answer:

"FAISS is a fast and lightweight vector search library. It is a good fit for this project because it supports efficient similarity search locally and can scale better than a naive Python loop over all chunks."

### Question: How do you stop hallucination?

Strong answer:

"I reduce hallucination by retrieving only the most relevant context, instructing the LLM to answer only from the provided chunks, and telling it to say it cannot find the information if the answer is missing. This does not remove hallucination completely, but it significantly reduces it."

### Question: What are the limitations?

Strong answer:

"The current system works well for text-based PDFs, but it is weaker on scanned documents, tables, and noisy extraction. It also depends heavily on retrieval quality. If the right chunk is not retrieved, the answer may be wrong even if the LLM is strong."

## 7. Questions the Candidate Should Be Able to Answer Without Hesitation

These are the minimum questions they should know cold.

1. What is RAG?
2. What is a vector embedding?
3. Why do we chunk documents?
4. What is FAISS used for?
5. What is the embedding model used in the project?
6. What does top-k retrieval mean?
7. Why is overlap used in chunking?
8. What does the LLM do after retrieval?
9. Why is `.env` used?
10. What happens if `GROQ_API_KEY` is missing?

## 8. Questions to Test Practical Debugging Skills

Ask these to see whether the candidate can actually work on the project.

1. The answer is wrong even though the document contains the information. How do you debug it?
2. Retrieval returns irrelevant chunks. What do you check first?
3. The PDF is scanned and text extraction is empty. What do you do?
4. Embedding generation is slow. How would you improve it?
5. FAISS search is returning poor results. What could be wrong?
6. The LLM is hallucinating even when the context is correct. Why might that happen?
7. The app crashes when a large PDF is uploaded. How do you fix it?
8. The same question gives inconsistent answers. What controls that behavior?
9. The app works locally but fails in deployment. What are the likely causes?
10. The API key is exposed accidentally. What is the immediate response?

## 9. Practical System Design Questions

These questions show whether the candidate can evolve the project beyond the current version.

1. How would you support multiple PDFs per user?
2. How would you let users search across an entire folder of documents?
3. How would you add citations or source references in answers?
4. How would you show the exact chunk used to answer the question?
5. How would you add chat memory across multiple documents?
6. How would you store embeddings in a database instead of RAM?
7. How would you make the system multi-tenant?
8. How would you add OCR for scanned PDFs?
9. How would you support document summarization in addition to Q&A?
10. How would you add evaluation benchmarks for retrieval and generation?

## 10. What The Candidate Needs To Learn To Truly Own This Project

This is the learning roadmap.

### A. RAG Fundamentals

Learn:

- What retrieval-augmented generation is.
- Why it is used.
- Where it fails.
- How retrieval and generation interact.

Must be able to explain:

- Semantic search.
- Chunking strategy.
- Prompt grounding.
- Hallucination control.

### B. Embeddings

Learn:

- What embeddings represent.
- Vector space similarity.
- Cosine similarity vs L2 distance.
- Embedding model selection.

Must be able to explain:

- Why semantic search works.
- Why one embedding model may be better than another.
- Why normalization can matter.

### C. Vector Databases and FAISS

Learn:

- Index types in FAISS.
- Flat search vs approximate search.
- How retrieval scales.
- How to persist indices.

Must be able to explain:

- Why FAISS is used here.
- What the tradeoffs are.
- When to use a managed vector database.

### D. Prompt Engineering

Learn:

- Instruction design.
- Grounding prompts.
- Answer constraints.
- Context formatting.

Must be able to explain:

- How to reduce hallucinations.
- How to make answers concise.
- How to force uncertainty when context is missing.

### E. Document Processing

Learn:

- PDF extraction limitations.
- OCR for scanned PDFs.
- Noise cleaning.
- Chunking strategies for different document types.

Must be able to explain:

- Why extracted text can be messy.
- Why some PDFs fail.
- How table-heavy documents are harder.

### F. Application Engineering

Learn:

- Streamlit session state.
- File upload flow.
- Error handling.
- Caching.
- Performance optimization.

Must be able to explain:

- How the UI state works.
- How the pipeline is reused.
- How the app handles repeated questions.

### G. Security

Learn:

- API key management.
- Environment variable usage.
- Git ignore hygiene.
- Secret scanning.

Must be able to explain:

- Why secrets must never be committed.
- What to do if a key leaks.
- How to make the repo safe to publish.

### H. Evaluation and Observability

Learn:

- How to test retrieval.
- How to test answer correctness.
- How to log safely.
- How to measure quality.

Must be able to explain:

- How to know whether the system is good.
- How to compare two retrieval strategies.
- How to detect regressions.

## 11. 7-Day Study Plan to Ace the Interview

### Day 1: Project Overview

- Explain the whole architecture out loud.
- Draw the pipeline from upload to answer.
- Review every source file.

### Day 2: RAG Fundamentals

- Study RAG concepts.
- Learn retrieval vs generation.
- Learn hallucination and grounding.

### Day 3: Embeddings and Vector Search

- Learn embeddings deeply.
- Understand cosine similarity and L2 distance.
- Study FAISS basics.

### Day 4: Chunking and PDF Processing

- Study chunk size, overlap, and text cleanup.
- Learn PDF limitations.
- Understand OCR and scanned documents.

### Day 5: Prompt Engineering and LLM Behavior

- Study prompt templates.
- Learn how temperature changes output.
- Practice explaining prompt constraints.

### Day 6: Debugging and Tradeoffs

- Practice failure scenarios.
- Learn how to improve retrieval quality.
- Review performance and security concerns.

### Day 7: Mock Interview

- Answer questions out loud.
- Time yourself.
- Practice short, precise answers.
- Prepare one architecture diagram and one tradeoff explanation.

## 12. Mock Interview Mode

If the interviewer asks for a quick explanation, use this structure:

1. What the project is.
2. What problem it solves.
3. How the pipeline works.
4. Why the choices were made.
5. What limitations remain.
6. What future improvements you would build.

Example short answer:

"This is a PDF question-answering system built with RAG. It extracts text from PDFs, chunks and embeds the text, stores vectors in FAISS, retrieves relevant chunks for a question, and sends them to Groq for grounded generation."

## 13. Red Flags in an Interview

The candidate should avoid:

- Memorizing buzzwords without understanding them.
- Saying "I used AI" without explaining the pipeline.
- Confusing embeddings with keywords.
- Claiming the LLM always knows the answer.
- Ignoring failure cases.
- Not knowing how the API key is stored.
- Not understanding why chunk overlap exists.

## 14. Strong Closing Statement for the Candidate

Use this kind of closing if asked to summarize the project:

"I built a document question-answering system using a RAG architecture. The system processes PDFs, creates embeddings, indexes them with FAISS, and uses a Groq LLM to generate grounded answers from retrieved context. I also handled session state in Streamlit and protected secrets using environment variables. If I were extending it, I would add OCR, citations, evaluation metrics, and persistent vector storage."

## 15. Final Advice

To truly own this project, the candidate should be able to explain not just what each file does, but why each design choice was made and what would break if that choice changed.

If they can answer the deep-dive questions in this guide, they will be ready for a strong technical interview on this project.
