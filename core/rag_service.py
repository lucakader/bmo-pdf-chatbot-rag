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
    
    @timing_decorator(operation_name="rag_query")
    def query(self, question: str) -> Dict[str, Any]:
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
            
            # Retrieve relevant documents
            logger.info(f"[{query_id}] Retrieving documents for query: {question[:50]}...")
            docs = self.retriever.retrieve(question)
            logger.info(f"[{query_id}] Retrieved {len(docs)} documents in {time.time() - start_time:.2f}s")
            
            # Format retrieved documents
            context_text, retrieval_id = self.retriever.format_retrieved_docs(docs)
            
            # Generate response
            logger.info(f"[{query_id}] Generating response...")
            response_start = time.time()
            response = self.rag_chain.invoke({
                "context": context_text, 
                "question": question
            })
            logger.info(f"[{query_id}] Generated response in {time.time() - response_start:.2f}s")
            
            # Check for hallucinations if enabled
            hallucination_result = None
            if self.check_hallucinations:
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
                "retrieved_docs": [{
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in docs],
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