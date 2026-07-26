# Enterprise-rag-knowledge-assistant

**Enterprise RAG Knowledge Assistant** — answers questions from internal documentation with retrieval + reranking + an explicit groundedness check, so answers stay tied to real source text instead of drifting into the model's own guesses.

## Summary
A standard RAG pipeline (embed → retrieve → generate) is fast but imprecise at the retrieval step. This assistant adds a cross-encoder reranking pass on top of bi-encoder retrieval to fix result ordering before generation, and a response-grounding check that verifies the generated answer actually cites a retrieved source before returning it to the user — falling back to "I couldn't find a well-grounded answer" otherwise.

## Architecture
```
Docs -> Chunk -> Embed -> Chroma index
Question -> Retrieve top-k (bi-encoder) -> Rerank top-n (cross-encoder) -> Generate (cited) -> Grounding check -> Answer
```

<img width="717" height="302" alt="Image" src="https://github.com/user-attachments/assets/25f465b6-969e-48f1-bd80-887358d54a83" />

## Stack
- LangChain
- Hugging Face embeddings (`all-MiniLM-L6-v2`) + cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`)
- ChromaDB
- OpenAI API

## Why it matters
Retrieval quality and groundedness are the two failure points that make or break an enterprise knowledge assistant — this project demonstrates both: narrowing noisy retrieval with a reranker, and refusing to answer when the evidence doesn't support it.

## Run
```bash
python enterprise_rag_knowledge_assistant.py
```
