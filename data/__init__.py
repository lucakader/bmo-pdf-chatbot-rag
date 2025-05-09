"""
Data module for PDF Chatbot RAG application.

This module handles document processing, embeddings generation,
and vector store interfaces.
"""

from data.document import DocumentProcessor
from data.vector_store import VectorStore, PineconeVectorStoreWrapper

__all__ = [
    'DocumentProcessor',
    'VectorStore',
    'PineconeVectorStoreWrapper'
]
