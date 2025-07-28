FROM python:3.10-slim

WORKDIR /app

# Install required OS packages
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy app and install dependencies
COPY app/ /app/
COPY app/requirement.txt /app/

RUN pip install --no-cache-dir -r requirement.txt

# Pre-download the model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

CMD ["python", "1b.py"]
