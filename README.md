# Approach Explanation

## Objective
The task is to extract and prioritize relevant information from a collection of travel-related PDF documents based on a given persona and job-to-be-done. The system should work entirely offline, within strict resource constraints: only CPU execution, model size under 1GB, and a total runtime within 60 seconds for processing 3–5 documents.

## Methodology

### 1. PDF Text Extraction
Each input PDF is parsed using the `PyMuPDF` (`fitz`) library. We extract text page by page and treat each page as a potential "section" if it contains sufficient text (over 100 characters). This allows us to isolate meaningful content across documents.

### 2. Text Embedding
To semantically compare the persona and task with the document sections, we use `sentence-transformers` with the `all-MiniLM-L6-v2` model. This model is compact (~90MB), fast, and provides strong semantic representation for sentence-level text. We generate dense vector embeddings for:
- The user query (persona + task)
- Each section’s full text

### 3. Relevance Ranking
Using cosine similarity between the user query vector and section embeddings, we score each section. The top 5 most relevant sections are selected and ranked by importance.

### 4. Subsection Analysis
For deeper insights, each of the top-ranked sections is split into sentences. Sentences are embedded and scored individually against the user query. The most relevant 5 sentences are extracted to form a refined summary of that section.

### 5. JSON Output
The system outputs a structured `output.json` file in the `/app/output` directory, containing:
- Metadata (persona, job, input documents, timestamp)
- Ranked relevant sections with their metadata
- Refined subsection analysis

## Design Highlights
- Runs fully on CPU
- Total processing time within 60 seconds for 3–5 documents
- All external models are pre-downloaded in Docker image (offline runtime)
- No dependencies on internet or external APIs

## Model Choice Justification
The `all-MiniLM-L6-v2` model was chosen because it balances accuracy and efficiency. It performs well on semantic similarity tasks and has a small memory footprint—ideal for constrained compute environments.

## Conclusion
This system provides a lightweight and robust pipeline for semantically understanding and extracting information from multiple documents. It is especially well-suited for offline, resource-constrained environments like document triage, travel planning, and summarization.

