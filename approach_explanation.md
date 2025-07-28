# 1B) Persona Driven Document Intelligence

# Approach Explanation

## Objective

The goal of this project is to extract the **most relevant sections** from a collection of PDFs to assist a persona in fulfilling a specific **job-to-be-done** (e.g., “Plan a 4-day trip for a group of 10 college friends” for persona "Travel Planner"). The final output is a structured JSON file containing:
- The top-ranked sections across documents
- Their titles and locations
- A refined summary from each section using extractive summarization

---

## Pipeline Overview

1. **Heading Detection and Section Segmentation**
2. **Sentence Embedding with Semantic Ranking**
3. **Extractive Summarization**
4. **JSON Output Generation**

---

## Step 1: Section Extraction using Font-Based Heading Detection

We use **PyMuPDF (fitz)** to parse the visual structure of the PDFs. To detect section titles:
- We analyze the **font sizes** and **frequency** of text styles across the document.
- Font sizes significantly **larger than the body text** are assumed to be section headings.
- Even if bold is not used, **relative font size** (and sometimes position on the page) provides strong cues.

This ensures the pipeline generalizes well to diverse PDF layouts — even those lacking tags or proper structure.

---

## Step 2: Semantic Ranking using `all-MiniLM-L6-v2`

We employ the [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model from **Sentence Transformers** to generate embeddings for each section and the user query (persona + task). This model is chosen because:

- **Small Size (≈80MB):** Ideal for CPU-only Docker deployment
- **Fast Inference:** Can encode hundreds of sections within seconds
- **Strong Performance:** Despite being compact, it achieves high semantic matching accuracy
- **Trained on Natural Language Inference and Semantic Textual Similarity**, making it ideal for text ranking tasks

We calculate **cosine similarity** between the query embedding and each section to identify the top 5 most relevant sections.

---

##  Step 3: Extractive Summarization

Within each top section, we split the text into individual sentences and re-score them using the same query embedding. We then select the top 5 most relevant sentences to form a **refined summary**.

This method provides summaries that are:
- Faithful to the original content (no hallucinations)
- Contextually relevant to the task
- Concise and informative

---

## Step 4: Output Generation

Finally, we structure the result as JSON containing:
- Document metadata
- Top 5 extracted sections (title, rank, page number)
- Subsection analysis with extractive summaries

This output can be used by any downstream UI, chatbot, or planning assistant.

---

## Why This Approach Works

This hybrid of **visual PDF analysis + sentence-transformer semantic ranking** gives us the best of both worlds:
- Human-like heading extraction without needing OCR
- Deep contextual understanding without expensive LLM inference


# The project is tested on sample input/output provided in the github link : https://github.com/jhaaj08/Adobe-India-Hackathon25.git


## Run Instructions
```sh
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
```

