from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain.schema import Document
import os
import logging
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve documents similar to the query."""
        pass
    
    @abstractmethod
    def as_retriever(self, **kwargs):
        """Return the vector store as a retriever."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get stats about the vector store."""
        pass
    
    @classmethod
    @abstractmethod
    def from_documents(cls, documents: List[Document], embeddings, **kwargs):
        """Create a vector store from documents."""
        pass

class PineconeVectorStoreWrapper(VectorStore):
    """Pinecone implementation of vector store with additional functionality."""
    
    def __init__(self, index_name: str, embedding_service=None, text_key="text"):
        """Initialize Pinecone vector store."""
        self.index_name = index_name
        
        # Initialize embedding service if not provided
        if embedding_service is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.embedding_service = OpenAIEmbeddings(api_key=api_key)
        else:
            self.embedding_service = embedding_service
            
        # Initialize Pinecone client
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        self.pc = Pinecone(api_key=api_key)
        
        # Create underlying vector store
        self.vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=self.embedding_service,
            text_key=text_key
        )
        
        logger.info(f"Initialized Pinecone vector store with index: {index_name}")
        
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to Pinecone."""
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to Pinecone index {self.index_name}")
        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {str(e)}")
            raise
        
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve documents from Pinecone."""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(docs)} documents from Pinecone for query: {query[:50]}...")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents from Pinecone: {str(e)}")
            raise
    
    def as_retriever(self, **kwargs):
        """Return the vector store as a retriever."""
        return self.vector_store.as_retriever(**kwargs)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get stats about the Pinecone index."""
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            return {
                "dimension": stats.get("dimension"),
                "namespaces": stats.get("namespaces", {}),
                "total_vector_count": stats.get("total_vector_count", 0)
            }
        except Exception as e:
            logger.error(f"Error getting stats from Pinecone: {str(e)}")
            return {"error": str(e)}
    
    @classmethod
    def from_documents(cls, documents: List[Document], embeddings, **kwargs):
        """Create a vector store from documents."""
        index_name = kwargs.get("index_name")
        if not index_name:
            raise ValueError("index_name is required")
            
        text_key = kwargs.get("text_key", "text")
        
        # Create the underlying vector store
        PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name,
            text_key=text_key
        )
        
        # Return our wrapper instance
        return cls(index_name=index_name, embedding_service=embeddings, text_key=text_key) 