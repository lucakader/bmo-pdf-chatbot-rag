#import Essential dependencies
import os
import sys
import time
import hashlib
import logging
from typing import List, Optional, Tuple, Union
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import signal
import backoff
import argparse

# Import config module
import config

# Add the current directory to the path
sys.path.insert(0, os.getcwd())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Timeout handler
class TimeoutError(Exception):
    """Exception raised when an operation times out."""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Vector store initialization timed out")

# Load environment variables
load_dotenv()

# Retry decorator for API calls
@backoff.on_exception(backoff.expo, 
                     (Exception,), 
                     max_tries=5,
                     max_time=300)
def api_call_with_retry(func, *args, **kwargs):
    """Execute a function with exponential backoff retry logic."""
    return func(*args, **kwargs)

def create_pinecone_index(pc: Pinecone, index_name: str, dimension: int = config.VECTOR_DIMENSION) -> bool:
    """
    Create a Pinecone index if it doesn't exist.
    
    Args:
        pc: Pinecone client instance
        index_name: Name of the index to create
        dimension: Dimension of vectors to store
        
    Returns:
        bool: True if the index was created or already exists, False otherwise
    """
    try:
        # Check if index exists first
        existing_indexes = pc.list_indexes().names()
        
        if index_name in existing_indexes:
            logger.info(f"Index {index_name} already exists, skipping creation")
            return True
            
        logger.info(f"Creating Pinecone index: {index_name}...")
        # Create serverless spec
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # Free tier typically available in us-east-1
        )
        
        api_call_with_retry(
            pc.create_index,
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=spec
        )
        
        # Wait for index to initialize with proper timeout
        start_time = time.time()
        max_wait_time = 300  # 5 minutes timeout
        check_interval = 10   # Check every 10 seconds
        
        while (time.time() - start_time) < max_wait_time:
            try:
                # Check if index is ready
                indexes = pc.list_indexes().names()
                if index_name in indexes:
                    logger.info(f"Index {index_name} is ready")
                    return True
            except Exception as e:
                logger.warning(f"Index not ready yet, waiting... ({str(e)})")
            
            # Calculate remaining time and log progress
            elapsed = time.time() - start_time
            remaining = max_wait_time - elapsed
            wait_count = int(elapsed / check_interval)
            logger.info(f"Waiting for index to be ready... ({wait_count} checks, {int(remaining)}s remaining)")
            
            # Sleep with check interval
            time.sleep(check_interval)
        
        # If we get here, timed out
        logger.error(f"Timed out waiting for index {index_name} to be ready after {max_wait_time} seconds")
        raise TimeoutError("Timed out waiting for index to be ready")
    
    except Exception as e:
        logger.error(f"Error creating index: {str(e)}")
        raise

def process_document(file_path: str) -> List:
    """
    Process a PDF document into text chunks.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List: Chunked document splits
    """
    try:
        logger.info(f"Loading PDF from {file_path}...")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        logger.info(f"Loaded {len(docs)} pages, splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, 
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        logger.info(f"Created {len(splits)} text chunks")
        
        return splits
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

def init_pinecone():
    """Initialize Pinecone client."""
    from pinecone import Pinecone
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    return pc

def init_document_processor():
    """Initialize document processor."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    class DocumentProcessor:
        def load_and_split(self, pdf_path):
            logger.info(f"Loading PDF from {pdf_path}...")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} pages, splitting text into chunks...")
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_documents(docs)
            logger.info(f"Created {len(chunks)} text chunks")
            return chunks
    
    return DocumentProcessor()

def create_vector_store(docs, custom_index_name=None):
    """Create vector store from documents."""
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    import hashlib
    
    logger.info("Creating vector store in Pinecone...")
    
    # Generate index name if not provided
    if not custom_index_name:
        # Create a hash based on the first document's content for consistency
        content_hash = hashlib.md5(docs[0].page_content.encode()).hexdigest()[:10]
        index_name = f"pdf-chatbot-{content_hash}"
    else:
        index_name = custom_index_name
    
    # Initialize embeddings
    api_key = os.environ.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    # Create or update vector store
    pc = init_pinecone()
    if index_name not in pc.list_indexes().names():
        # Create new index
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embedding dimension
            metric="cosine"
        )
    
    # Add documents to index
    PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=index_name
    )
    
    return index_name

def save_chunks_for_bm25(docs):
    """Save document chunks for BM25 search."""
    output_path = "data/document_chunks.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for i, chunk in enumerate(docs):
            f.write(f"--- Chunk {i+1} ---\n")
            f.write(chunk.page_content)
            f.write("\n\n")
    
    logger.info(f"Wrote {len(docs)} chunks to {output_path} for BM25 search")

def signal_handler(signum, frame):
    """Handle interrupt signals."""
    logger.info("Received interrupt signal, exiting gracefully...")
    sys.exit(0)

def main():
    """Main function to initialize the vector store from a PDF."""
    load_dotenv()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    start_time = time.time()
    
    # Check required environment variables
    if not os.environ.get('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    if not os.environ.get('PINECONE_API_KEY'):
        logger.error("PINECONE_API_KEY environment variable not set")
        sys.exit(1)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Initialize vector store from PDF')
    parser.add_argument('--pdf-path', type=str, help='Path to PDF file')
    parser.add_argument('--index-name', type=str, help='Pinecone index name')
    args = parser.parse_args()
    
    # Initialize Pinecone
    logger.info("Initializing Pinecone...")
    
    # Get PDF path from argument, environment variable, or default
    pdf_path = args.pdf_path or os.environ.get('PDF_PATH', 'random machine learing pdf.pdf')
    logger.info(f"Using PDF at path: {pdf_path}")
    
    # Create vector store with custom index name if provided
    index_name = args.index_name or os.environ.get('PINECONE_INDEX_NAME', None)
    
    # If index_name is provided, check if it exists
    if index_name:
        try:
            pc = init_pinecone()
            if index_name in pc.list_indexes().names():
                logger.info(f"Index {index_name} already exists, using it")
                # Save index name to file
                os.makedirs('vectorstore', exist_ok=True)
                with open('vectorstore/index_name.txt', 'w') as f:
                    f.write(index_name)
                logger.info(f"Index name saved to vectorstore/index_name.txt")
                return
        except Exception as e:
            logger.warning(f"Error checking index: {e}")
    
    # Process document
    try:
        # Initialize document processor
        doc_processor = init_document_processor()
        
        # Load and process PDF
        docs = doc_processor.load_and_split(pdf_path)
        
        # Create vector store
        index_name = create_vector_store(docs, index_name)
        logger.info(f"Vector store created successfully in Pinecone index: {index_name}")
        
        # Save chunks for BM25 search
        save_chunks_for_bm25(docs)
        
        # Save index name to file for later use
        os.makedirs('vectorstore', exist_ok=True)
        with open('vectorstore/index_name.txt', 'w') as f:
            f.write(index_name)
        logger.info(f"Index name saved to vectorstore/index_name.txt")
        
        end_time = time.time()
        logger.info(f"Vector store initialization completed in {end_time - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(2)

if __name__=="__main__":
    sys.exit(main() or 0)