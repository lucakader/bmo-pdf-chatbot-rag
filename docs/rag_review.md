# RAG Implementation Review

This project already has the right building blocks for a resume-grade RAG demo: PDF ingestion, chunking, embeddings, vector retrieval, optional BM25 hybrid retrieval, source-aware prompting, Streamlit UI, Docker/Kubernetes deployment, and Prometheus metrics.

## Flaws Found

- The RAG chain passed the whole input dictionary into both `context` and `question`, which could produce prompts with malformed context/question values.
- Streamlit sidebar controls were not applied to the query path, so retrieval count, hybrid search, reranking, hallucination checks, confidence threshold, and temperature looked interactive but did not affect answers.
- BM25 chunks were stored as plain text, so keyword retrieval lost source and page metadata needed for trustworthy citations.
- The app created an empty BM25 file when chunks were missing, which made hybrid retrieval appear enabled even when there was no keyword corpus.
- The project had no automated tests around the core RAG contract.
- README positioning was generic and did not clearly explain why the implementation is credible as a RAG portfolio project.

## Updates Made

- Fixed RAG chain input mapping so `context` and `question` are routed explicitly.
- Wired UI options through to `RAGService.query()` and rebuilt retriever settings at runtime when those options change.
- Added a no-context fallback so the app does not call the LLM when retrieval returns no supporting chunks.
- Added a retrieved-sources panel in the UI so users can inspect the snippets and settings behind each answer.
- Preserved BM25 source metadata by saving generated chunks as JSONL and loading legacy plain-text chunks as a fallback.
- Avoided creating fake empty BM25 corpora at startup.
- Added offline unit tests with fake vector store and fake LLM components.

## Recommended Next Improvements

- Add a small evaluation set with expected answers and cited pages, then track retrieval hit rate and grounded-answer quality.
- Add a local vector-store option for demo mode so reviewers can run the project without provisioning Pinecone.
- Replace the single sample PDF assumption with upload/index/select document workflows.
- Add screenshots or a short GIF to the README showing the UI, citations, and retrieved chunks.
- Modernize Pinecone initialization in `data/vector_loader.py` if the project upgrades fully to the current Pinecone SDK.
