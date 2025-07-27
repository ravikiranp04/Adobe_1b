# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY app/1b.py /app/
COPY app/requirement.txt /app/

RUN pip install --no-cache-dir -r requirement.txt

# Pre-download model for offline runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

CMD ["python", "1b.py"]
