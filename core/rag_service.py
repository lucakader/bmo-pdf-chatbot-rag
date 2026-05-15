from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import uuid
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from data.vector_store import VectorStore
from core.llm import LLMProvider
from core.retrieval import EnhancedRetriever
from core.validation import ResponseValidator, HallucinationCheck
from monitoring.metrics import timing_decorator
import config

logger = logging.getLogger(__name__)

class RAGService:
    """Service for Retrieval-Augmented Generation (RAG)."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        use_hybrid_search: bool = config.USE_HYBRID_SEARCH,
        use_reranker: bool = config.RERANKER_ENABLED,
        check_hallucinations: bool = config.HALLUCINATION_CHECK_ENABLED,
        confidence_threshold: float = 0.6,
        vector_weight: float = config.VECTOR_WEIGHT,
        bm25_weight: float = config.BM25_WEIGHT,
        retrieval_k: int = config.RETRIEVAL_K,
        bm25_docs_path: Optional[str] = config.BM25_DOCS_PATH
    ):
        """Initialize RAG service."""
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.use_hybrid_search = use_hybrid_search
        self.use_reranker = use_reranker
        self.check_hallucinations = check_hallucinations
        self.confidence_threshold = confidence_threshold
        
        # Initialize retriever
        self.retriever = EnhancedRetriever(
            vector_store=vector_store,
            llm=llm_provider.get_llm(),
            use_hybrid_search=use_hybrid_search,
            use_reranker=use_reranker,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            retrieval_k=retrieval_k
        )
        
        # Load BM25 documents if provided and hybrid search is enabled
        if use_hybrid_search and bm25_docs_path:
            self.retriever.load_documents_for_bm25(bm25_docs_path)
        
        # Initialize response validator
        self.validator = ResponseValidator(
            llm=llm_provider.get_llm(),
            confidence_threshold=confidence_threshold
        )
        
        # Load RAG prompt
        self.prompt = self._load_rag_prompt()
        
        # Create RAG chain
        self.rag_chain = llm_provider.create_rag_chain(self.prompt)
        
        logger.info(f"Initialized RAG service with: hybrid_search={use_hybrid_search}, "
                   f"reranker={use_reranker}, hallucination_check={check_hallucinations}")
        
    def _load_rag_prompt(self):
        """Load the RAG prompt."""
        # Create a custom prompt that includes source attribution
        return ChatPromptTemplate.from_template("""
        You are a helpful assistant answering questions about a document.

        Given the context information below, answer the query.
        
        If you don't know the answer based ONLY on the context provided, say "I don't have enough information to answer this question."
        Keep your answer detailed but concise. Provide specific quotes or page numbers when possible.
        
        Always include a "Sources:" section at the end of your answer that lists the specific sources or chunks used.
        
        Context:
        {context}
        
        Query: {question}
        """)

    def _current_retrieval_settings(self) -> Dict[str, Any]:
        """Return retrieval settings used for this query."""
        return {
            "use_hybrid_search": self.retriever.use_hybrid_search,
            "use_reranker": self.retriever.use_reranker,
            "vector_weight": self.retriever.vector_weight,
            "bm25_weight": self.retriever.bm25_weight,
            "retrieval_k": self.retriever.retrieval_k,
        }

    def _serialize_retrieved_docs(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Serialize retrieved documents for debugging, UI display, and tests."""
        serialized_docs = []
        for rank, doc in enumerate(docs, start=1):
            content = doc.page_content.strip()
            serialized_docs.append({
                "rank": rank,
                "content": content,
                "preview": content[:300],
                "metadata": doc.metadata,
            })
        return serialized_docs
    
    @timing_decorator(operation_name="rag_query")
    def query(
        self,
        question: str,
        use_hybrid_search: Optional[bool] = None,
        use_reranker: Optional[bool] = None,
        vector_weight: Optional[float] = None,
        check_for_hallucinations: Optional[bool] = None,
        confidence_threshold: Optional[float] = None,
        temperature: Optional[float] = None,
        retrieval_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a question using RAG.
        
        Args:
            question: The user's question
            
        Returns:
            Dict[str, Any]: Response with metadata
        """
        try:
            start_time = time.time()
            query_id = str(uuid.uuid4())
            question = question.strip()

            if not question:
                return {
                    "query_id": query_id,
                    "question": question,
                    "response": "Please ask a question about the indexed document.",
                    "retrieved_docs": [],
                    "retrieval_settings": self._current_retrieval_settings(),
                    "processing_time": time.time() - start_time,
                    "validation_info": {"warning": "Empty question"},
                }
            
            bm25_weight = None
            if vector_weight is not None:
                bm25_weight = max(0.0, min(1.0, 1.0 - vector_weight))

            self.retriever.configure(
                use_hybrid_search=use_hybrid_search,
                use_reranker=use_reranker,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                retrieval_k=retrieval_k
            )
            hallucination_check_enabled = (
                self.check_hallucinations
                if check_for_hallucinations is None
                else check_for_hallucinations
            )
            if confidence_threshold is not None:
                self.confidence_threshold = confidence_threshold
                self.validator.confidence_threshold = confidence_threshold

            # Retrieve relevant documents
            logger.info(f"[{query_id}] Retrieving documents for query: {question[:50]}...")
            docs = self.retriever.retrieve(question)
            logger.info(f"[{query_id}] Retrieved {len(docs)} documents in {time.time() - start_time:.2f}s")

            retrieved_docs = self._serialize_retrieved_docs(docs)
            retrieval_settings = self._current_retrieval_settings()

            if not docs:
                return {
                    "query_id": query_id,
                    "question": question,
                    "response": "I don't have enough information in the indexed document to answer this question.",
                    "retrieved_docs": [],
                    "retrieval_settings": retrieval_settings,
                    "processing_time": time.time() - start_time,
                    "validation_info": {
                        "has_citations": False,
                        "warning": "No retrieved context",
                    },
                }
            
            # Format retrieved documents
            context_text, retrieval_id = self.retriever.format_retrieved_docs(docs)
            
            # Generate response
            logger.info(f"[{query_id}] Generating response...")
            response_start = time.time()
            rag_chain = (
                self.llm_provider.create_rag_chain(self.prompt, temperature=temperature)
                if temperature is not None
                else self.rag_chain
            )
            response = rag_chain.invoke({
                "context": context_text, 
                "question": question
            })
            logger.info(f"[{query_id}] Generated response in {time.time() - response_start:.2f}s")
            
            # Check for hallucinations if enabled
            hallucination_result = None
            if hallucination_check_enabled:
                logger.info(f"[{query_id}] Checking for hallucinations...")
                hallucination_result = self.validator.check_hallucination(response, context_text, question)
                
            # Process and validate the response
            if hallucination_result:
                # Validate response
                validated_response, validation_info = self.validator.validate_response(
                    response, context_text, question, hallucination_result
                )
                
                # Use fallback response if confidence is too low
                if hallucination_result.confidence_score < self.confidence_threshold:
                    logger.warning(f"[{query_id}] Low confidence ({hallucination_result.confidence_score}) "
                                  f"below threshold ({self.confidence_threshold}), using fallback")
                    validated_response = self.validator.generate_fallback_response(
                        question, 
                        hallucination_result.confidence_score, 
                        hallucination_result.reasoning
                    )
                    validation_info = {
                        'has_citations': False,
                        'warning': 'Low confidence response',
                        'confidence': hallucination_result.confidence_score,
                        'hallucination_check': {
                            'is_hallucination': hallucination_result.is_hallucination,
                            'confidence_score': hallucination_result.confidence_score,
                            'reasoning': hallucination_result.reasoning
                        }
                    }
                
                final_response = validated_response
            else:
                # No hallucination check, just validate the response for citations
                final_response, validation_info = self.validator.validate_response(
                    response, context_text, question
                )
            
            # Build response with metadata
            result = {
                "query_id": query_id,
                "question": question,
                "response": final_response,
                "retrieved_docs": retrieved_docs,
                "retrieval_settings": retrieval_settings,
                "processing_time": time.time() - start_time,
                "validation_info": validation_info
            }
            
            # Add hallucination check result if available
            if hallucination_result:
                result["hallucination_check"] = {
                    "is_hallucination": hallucination_result.is_hallucination,
                    "confidence_score": hallucination_result.confidence_score,
                    "reasoning": hallucination_result.reasoning,
                    "verified_claims": hallucination_result.verified_claims,
                    "unverified_claims": hallucination_result.unverified_claims
                }
            
            return result
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "query_id": str(uuid.uuid4()),
                "question": question,
                "error": str(e),
                "response": "I encountered an error while processing your question. Please try again."
            } 
