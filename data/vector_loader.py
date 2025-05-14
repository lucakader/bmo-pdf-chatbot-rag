#!/usr/bin/env python3
"""
Vector store initialization script for Kubernetes deployment.
"""
import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

# Import required modules
try:
    from data.document import DocumentProcessor
    from langchain_openai import OpenAIEmbeddings
    import pinecone
    import config
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure all dependencies are installed.")
    sys.exit(1)

def initialize_vector_store(pdf_path, index_name, use_existing=False):
    """Initialize the vector store with document embeddings."""
    logger.info(f"Initializing vector store from PDF: {pdf_path}")
    
    # Create document processor
    processor = DocumentProcessor()
    docs = processor.process_pdf(pdf_path)
    logger.info(f"Processed {len(docs)} document chunks")
    
    # Get embeddings service
    embeddings = OpenAIEmbeddings(api_key=os.environ.get('OPENAI_API_KEY'))
    
    # Initialize Pinecone
    api_key = os.environ.get('PINECONE_API_KEY')
    environment = os.environ.get('PINECONE_ENVIRONMENT')
    
    if not api_key or not environment:
        logger.error("Pinecone API key or environment not set")
        return False
    
    logger.info(f"Using Pinecone index: {index_name}")
    
    try:
        # Initialize Pinecone client
        pinecone.init(api_key=api_key, environment=environment)
        
        # Check if index exists
        indexes = pinecone.list_indexes()
        
        if index_name not in indexes:
            logger.error(f"Index {index_name} does not exist. Please create it in the Pinecone UI first.")
            return False
        
        logger.info(f"Using existing Pinecone index: {index_name}")
        
        # Connect to the index
        index = pinecone.Index(index_name)
        
        # Create embeddings for documents
        logger.info("Creating embeddings for documents...")
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        
        # Get embeddings
        embeddings_model = OpenAIEmbeddings()
        embeds = embeddings_model.embed_documents(texts)
        
        # Prepare for upsert
        vectors_to_upsert = []
        for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeds)):
            vectors_to_upsert.append({
                "id": f"doc_{i}",
                "values": embedding,
                "metadata": {**metadata, "text": text}
            })
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i+batch_size]
            index.upsert(vectors=batch)
            logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1}")
        
        # Save chunks for BM25
        processor.save_chunks_for_bm25(docs, config.BM25_DOCS_PATH)
        logger.info(f"Saved document chunks for BM25 to {config.BM25_DOCS_PATH}")
        
        logger.info("Vector store initialization complete")
        return True
    
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Initialize vector store for PDF chatbot")
    parser.add_argument('--index-name', type=str, help='Name of the Pinecone index to use')
    parser.add_argument('--pdf-path', type=str, help='Path to the PDF file')
    parser.add_argument('--use-existing', action='store_true', help='Use existing index without creating vectors')
    
    args = parser.parse_args()
    
    # Get args or environment variables
    index_name = args.index_name or os.environ.get('PINECONE_INDEX_NAME')
    pdf_path = args.pdf_path or os.environ.get('PDF_PATH', '/app/data/random_machine_learning_pdf.pdf')
    
    if not index_name:
        index_name = f'pdf-chatbot-{int(time.time())}'
        logger.info(f"No index name provided, using generated name: {index_name}")
    else:
        logger.info(f"Using index name from environment: {index_name}")
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found at {pdf_path}")
        sys.exit(1)
    
    # Initialize vector store
    success = initialize_vector_store(pdf_path, index_name, args.use_existing)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 