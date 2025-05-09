from typing import List, Dict, Any, Optional, Union
from langchain.schema import Document
import logging
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

logger = logging.getLogger(__name__)

class EnhancedRetriever:
    """Enhanced retriever with hybrid search and reranking capabilities."""
    
    def __init__(
        self, 
        vector_store,
        llm=None,
        use_hybrid_search: bool = True,
        use_reranker: bool = True,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        retrieval_k: int = 5,
        documents=None
    ):
        """Initialize the enhanced retriever."""
        self.vector_store = vector_store
        self.llm = llm
        self.use_hybrid_search = use_hybrid_search
        self.use_reranker = use_reranker
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.retrieval_k = retrieval_k
        self.documents = documents
        
        # Set up the base retriever
        self._create_base_retriever()
        
    def _create_base_retriever(self):
        """Create the base retriever based on configuration."""
        # Set up the vector store retriever
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.retrieval_k})
        
        # Create hybrid search if enabled
        if self.use_hybrid_search and self.documents:
            self.base_retriever = self._create_hybrid_retriever(vector_retriever)
        else:
            self.base_retriever = vector_retriever
            
        # Add reranker if enabled
        if self.use_reranker and self.llm:
            self.retriever = self._create_reranker(self.base_retriever)
        else:
            self.retriever = self.base_retriever
            
        logger.info(f"Created retriever: hybrid_search={self.use_hybrid_search}, reranker={self.use_reranker}")
    
    def _create_hybrid_retriever(self, vector_retriever):
        """Create a hybrid retriever combining vector search with BM25."""
        try:
            # Create a BM25 retriever
            logger.info(f"Creating BM25 retriever with {len(self.documents)} documents")
            bm25_retriever = BM25Retriever.from_documents(self.documents)
            bm25_retriever.k = self.retrieval_k
            
            # Create the ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[self.vector_weight, self.bm25_weight]
            )
            
            logger.info(f"Created hybrid retriever with weights: vector={self.vector_weight}, bm25={self.bm25_weight}")
            return ensemble_retriever
        except Exception as e:
            logger.error(f"Error creating hybrid retriever: {str(e)}")
            logger.warning("Falling back to vector store only retrieval")
            return vector_retriever
    
    def _create_reranker(self, retriever):
        """Create a contextual compression retriever for reranking."""
        try:
            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_retriever=retriever,
                base_compressor=compressor
            )
            logger.info("Created reranker using LLM-based contextual compression")
            return compression_retriever
        except Exception as e:
            logger.error(f"Error creating reranker: {str(e)}")
            logger.warning("Falling back to base retriever without reranking")
            return retriever
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve documents relevant to the query."""
        try:
            logger.info(f"Retrieving documents for query: {query[:50]}...")
            docs = self.retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
            
    def format_retrieved_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents with source information."""
        # Generate a unique retrieval ID
        retrieval_id = str(uuid.uuid4())
        
        # Format with source identifiers
        formatted_text = ""
        for i, doc in enumerate(docs):
            # Add source identifier and metadata if available
            source_info = f"[Source {i+1}"
            if hasattr(doc, 'metadata') and doc.metadata:
                if 'page' in doc.metadata:
                    source_info += f", Page {doc.metadata['page']}"
                if 'source' in doc.metadata:
                    source_info += f", {doc.metadata['source'].split('/')[-1]}"
            source_info += "]"
            
            # Add content with source tag
            formatted_text += f"{source_info}\n{doc.page_content}\n\n"
            
        return formatted_text, retrieval_id
        
    def load_documents_for_bm25(self, file_path: str):
        """Load text chunks from file for BM25 retrieval."""
        try:
            with open(file_path, "r") as f:
                text = f.read()
                
            # Split into chunks for BM25
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            self.documents = text_splitter.create_documents([text])
            logger.info(f"Loaded {len(self.documents)} document chunks for BM25 retrieval")
            
            # Recreate the base retriever with new documents
            self._create_base_retriever()
            
            return True
        except Exception as e:
            logger.error(f"Error loading documents for BM25: {str(e)}")
            return False 