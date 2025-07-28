import os
import fitz  # PyMuPDF
import json
import datetime
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

model = SentenceTransformer("all-MiniLM-L6-v2")

def is_bold(span):
    return "bold" in span.get("font", "").lower() or (span["flags"] & 16)

def extract_headings(pdf_path):
    doc = fitz.open(pdf_path)
    font_sizes = defaultdict(int)
    headings = []

    # Pass 1: collect font sizes
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = round(span["size"], 1)
                    font_sizes[size] += 1

    if not font_sizes:
        return []

    # Determine body font size as most frequent
    body_size = max(font_sizes.items(), key=lambda x: x[1])[0]
    heading_sizes = [size for size in font_sizes if size > body_size]

    if not heading_sizes:
        heading_sizes = sorted(font_sizes.keys(), reverse=True)[:2]  # fallback

    # Map sizes to H1-H4 levels
    sorted_sizes = sorted(set(heading_sizes), reverse=True)
    font_level_map = {sz: f"H{idx+1}" for idx, sz in enumerate(sorted_sizes[:4])}

    for page_num, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                line_text = ""
                sizes = []
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    size = round(span["size"], 1)
                    line_text += text + " "
                    sizes.append(size)

                if not line_text.strip() or not sizes:
                    continue

                avg_size = round(sum(sizes) / len(sizes), 1)
                if avg_size in font_level_map:
                    headings.append({
                        "level": font_level_map[avg_size],
                        "text": line_text.strip(),
                        "page": page_num + 1
                    })

    return headings

def extract_sections_with_text(pdf_path, headings):
    doc = fitz.open(pdf_path)
    sections = []
    for heading in headings:
        page = doc[heading["page"] - 1]
        text = page.get_text()
        if len(text.strip()) < 100:
            continue
        sections.append({
            "document": os.path.basename(pdf_path),
            "section_title": heading["text"],
            "text": text,
            "page_number": heading["page"]
        })
    return sections

def generate_embeddings(texts):
    return model.encode(texts, convert_to_tensor=False)

def rank_sections(sections, persona, job):
    query = f"{persona} - {job}"
    query_embedding = generate_embeddings([query])[0]
    section_texts = [s["text"] for s in sections]
    section_embeddings = generate_embeddings(section_texts)

    similarities = cosine_similarity([query_embedding], section_embeddings)[0]
    for i, score in enumerate(similarities):
        sections[i]["score"] = float(score)

    ranked = sorted(sections, key=lambda x: x["score"], reverse=True)[:5]
    for rank, s in enumerate(ranked, start=1):
        s["importance_rank"] = rank

    return ranked, query_embedding

def analyze_subsections(ranked_sections, query_embedding):
    refined = []
    for s in ranked_sections:
        sentences = [sent.strip() for sent in s["text"].split(". ") if len(sent.strip()) > 20]
        if not sentences:
            continue
        sent_embeddings = generate_embeddings(sentences)
        scores = cosine_similarity([query_embedding], sent_embeddings)[0]
        top_sentences = [sentences[i] for i in np.argsort(scores)[-5:]]
        refined_text = ". ".join(top_sentences).strip()
        refined.append({
            "document": s["document"],
            "refined_text": refined_text,
            "page_number": s["page_number"]
        })
    return refined

def generate_output(input_json_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    input_files = [os.path.join(INPUT_FOLDER, doc["filename"]) for doc in data["documents"]]
    persona = data["persona"]["role"]
    job = data["job_to_be_done"]["task"]

    all_sections = []
    for pdf_path in input_files:
        if not os.path.exists(pdf_path):
            print(f"❌ Missing: {pdf_path}")
            continue
        headings = extract_headings(pdf_path)
        if not headings:
            print(f"⚠️  No headings found in {pdf_path}")
        extracted = extract_sections_with_text(pdf_path, headings)
        all_sections.extend(extracted)

    if not all_sections:
        raise ValueError("❌ No sections extracted from any PDFs.")

    ranked_sections, query_embedding = rank_sections(all_sections, persona, job)
    subsection_analysis = analyze_subsections(ranked_sections, query_embedding)

    output = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in data["documents"]],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": [{
            "document": s["document"],
            "section_title": s["section_title"],
            "importance_rank": s["importance_rank"],
            "page_number": s["page_number"]
        } for s in ranked_sections],
        "subsection_analysis": subsection_analysis
    }

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER, "output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    
    return output

if __name__ == "__main__":
    input_json_path = os.path.join(INPUT_FOLDER, "input.json")
    generate_output(input_json_path)
