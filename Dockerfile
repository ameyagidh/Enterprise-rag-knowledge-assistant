FROM python:3.11-slim

# System deps for docx2txt / building some wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

# Install the package first (without dev extras) so layer caching keeps this
# fast on rebuilds that only touch application code below.
RUN pip install --no-cache-dir .

# Pre-download the embedding + reranker model weights at build time, so
# `docker compose up` starts immediately and works even with restricted
# egress at runtime. This is the deliberate tradeoff: slower image build,
# fast/offline-capable container start.
RUN python -c "\
from langchain_huggingface import HuggingFaceEmbeddings; \
from sentence_transformers import CrossEncoder; \
HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY ui ./ui
COPY knowledge_base ./knowledge_base
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

EXPOSE 8000 8501

ENTRYPOINT ["./docker-entrypoint.sh"]
