import fitz
import os
import json
import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load compact model
model = SentenceTransformer("all-MiniLM-L6-v2")

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

def extract_text_sections(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if len(text.strip()) > 100:
            sections.append({
                "document": os.path.basename(pdf_path),
                "text": text,
                "page_number": page_num,
                "section_title": f"Page {page_num} Section"
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
        sentences = s["text"].split(". ")
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

def generate_output_from_input_file(input_file_path):
    with open(input_file_path, "r") as f:
        data = json.load(f)

    input_docs = [os.path.join(INPUT_FOLDER, doc["filename"]) for doc in data["documents"]]
    persona = data["persona"]["role"]
    job = data["job_to_be_done"]["task"]

    all_sections = []
    for doc_path in input_docs:
        if os.path.exists(doc_path):
            all_sections.extend(extract_text_sections(doc_path))
        else:
            print(f"[Warning] File not found: {doc_path}")

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
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"✅ Output saved to {output_path}")
    return output

if __name__ == "__main__":
    input_json_path = os.path.join(INPUT_FOLDER, "input.json")
    generate_output_from_input_file(input_json_path)
