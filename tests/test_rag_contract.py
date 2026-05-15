import tempfile
import unittest

from langchain_core.runnables import RunnableLambda
from langchain.schema import Document

from core.rag_service import RAGService
from core.retrieval import EnhancedRetriever
from data.document import DocumentProcessor


class FakeRetriever:
    def __init__(self, docs=None):
        self.search_kwargs = {}
        self.docs = docs if docs is not None else [
            Document(
                page_content="Relevant answer text",
                metadata={"source": "sample.pdf", "page": 2},
            )
        ]

    def get_relevant_documents(self, query):
        return self.docs


class FakeVectorStore:
    def __init__(self, docs=None):
        self.retriever = FakeRetriever(docs=docs)

    def as_retriever(self, **kwargs):
        self.retriever.search_kwargs = kwargs.get("search_kwargs", {})
        return self.retriever


class FakeLLM:
    def with_structured_output(self, output_class):
        return self


class FakeLLMProvider:
    def __init__(self):
        self.temperatures = []

    def get_llm(self, **kwargs):
        return FakeLLM()

    def create_rag_chain(self, prompt, temperature=None):
        self.temperatures.append(temperature)
        return RunnableLambda(
            lambda inputs: (
                f"Answer from {inputs['context']} for {inputs['question']}\n\n"
                "Sources:\n- Source 1, page 2"
            )
        )


class RAGContractTests(unittest.TestCase):
    def test_query_uses_runtime_options_and_preserves_sources(self):
        provider = FakeLLMProvider()
        service = RAGService(
            vector_store=FakeVectorStore(),
            llm_provider=provider,
            use_hybrid_search=False,
            use_reranker=False,
            check_hallucinations=False,
        )

        result = service.query(
            "What is covered?",
            use_hybrid_search=True,
            use_reranker=False,
            vector_weight=0.8,
            check_for_hallucinations=False,
            confidence_threshold=0.7,
            temperature=0.2,
            retrieval_k=3,
        )

        self.assertIn("Relevant answer text", result["response"])
        self.assertEqual(result["retrieved_docs"][0]["rank"], 1)
        self.assertEqual(result["retrieved_docs"][0]["preview"], "Relevant answer text")
        self.assertEqual(result["retrieved_docs"][0]["metadata"]["page"], 2)
        self.assertEqual(result["retrieval_settings"]["retrieval_k"], 3)
        self.assertEqual(service.retriever.vector_weight, 0.8)
        self.assertAlmostEqual(service.retriever.bm25_weight, 0.2)
        self.assertEqual(service.retriever.retrieval_k, 3)
        self.assertEqual(service.confidence_threshold, 0.7)
        self.assertIn(0.2, provider.temperatures)

    def test_query_returns_grounded_fallback_when_no_docs_are_retrieved(self):
        provider = FakeLLMProvider()
        service = RAGService(
            vector_store=FakeVectorStore(docs=[]),
            llm_provider=provider,
            use_hybrid_search=False,
            use_reranker=False,
            check_hallucinations=False,
        )

        result = service.query("What is not in the document?")

        self.assertIn("don't have enough information", result["response"])
        self.assertEqual(result["retrieved_docs"], [])
        self.assertEqual(result["validation_info"]["warning"], "No retrieved context")
        self.assertEqual(provider.temperatures, [None])

    def test_bm25_jsonl_round_trip_preserves_metadata(self):
        chunks = [
            Document(
                page_content="A chunk about retrieval.",
                metadata={"source": "guide.pdf", "page": 4},
            )
        ]

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl") as output:
            processor = DocumentProcessor()
            self.assertTrue(processor.save_chunks_for_bm25(chunks, output.name))

            retriever = EnhancedRetriever(
                vector_store=FakeVectorStore(),
                use_hybrid_search=False,
                use_reranker=False,
            )
            self.assertTrue(retriever.load_documents_for_bm25(output.name))

        self.assertEqual(retriever.documents[0].page_content, "A chunk about retrieval.")
        self.assertEqual(retriever.documents[0].metadata["source"], "guide.pdf")
        self.assertEqual(retriever.documents[0].metadata["page"], 4)


if __name__ == "__main__":
    unittest.main()
