import os
import sys
import logging
import streamlit as st
from dotenv import load_dotenv
import config

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from data.vector_store import PineconeVectorStoreWrapper
from data.document import DocumentProcessor
from core.llm import OpenAIProvider, CachedLLMProvider
from core.rag_service import RAGService
from app.ui.chat_ui import ChatUI
from monitoring.metrics import MetricsManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set page config as the first Streamlit command
st.set_page_config(page_title="PDF Chatbot", page_icon="📝", layout="wide")

def load_environment():
    """Load environment variables."""
    load_dotenv()
    
    # Validate environment variables
    is_valid, missing = config.validate_environment()
    if not is_valid:
        for var in missing:
            logger.error(f"Missing required environment variable: {var}")
            st.error(f"Missing required environment variable: {var}")
        st.stop()
        
    return is_valid

def initialize_metrics():
    """Initialize metrics server."""
    # Detect if we're running in Kubernetes
    is_k8s = os.environ.get('KUBERNETES_SERVICE_HOST') is not None
    logger.info(f"Running in Kubernetes environment: {is_k8s}")
    
    # Get metrics port from environment or use default
    metrics_port = int(os.environ.get('METRICS_PORT', config.METRICS_PORT))
    
    # Initialize metrics manager
    metrics_manager = MetricsManager(metrics_port=metrics_port, enable_metrics=config.METRICS_ENABLED)
    
    # Only start metrics server if we're not running in Kubernetes
    if not is_k8s:
        metrics_manager.start_metrics_server()
    else:
        logger.info("Running in Kubernetes - metrics will be handled by sidecar container")
        
    return metrics_manager

def initialize_vector_store():
    """Initialize the vector store."""
    try:
        # Read index name from file
        if not os.path.exists("vectorstore/index_name.txt"):
            logger.error("Vector store index name not found. Please run the application setup first.")
            st.error("Vector store index name not found. Please run the start.sh script first to initialize the vector store.")
            st.stop()
            
        with open("vectorstore/index_name.txt", "r") as f:
            index_name = f.read().strip()
        
        logger.info(f"Using Pinecone index: {index_name}")
        
        # Create vector store wrapper
        vector_store = PineconeVectorStoreWrapper(index_name=index_name)
        
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        st.error(f"Error initializing vector store: {str(e)}")
        st.stop()

def initialize_document_processor():
    """Initialize the document processor."""
    return DocumentProcessor(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

def initialize_llm_provider():
    """Initialize the LLM provider with caching."""
    # Create base OpenAI provider
    try:
        base_provider = OpenAIProvider(
            model_name=config.DEFAULT_LLM_MODEL,
            temperature=config.DEFAULT_LLM_TEMPERATURE
        )
        
        # Add caching if enabled
        if config.CACHE_EMBEDDINGS:
            return CachedLLMProvider(base_provider, cache_size=100)
        else:
            return base_provider
    except Exception as e:
        logger.error(f"Error initializing LLM provider: {str(e)}")
        st.error(f"Error initializing LLM provider: {str(e)}")
        st.stop()

def initialize_rag_service(vector_store, llm_provider):
    """Initialize the RAG service."""
    try:
        # Check if BM25 docs exist
        bm25_docs_path = "data/document_chunks.txt"
        if not os.path.exists(bm25_docs_path):
            logger.warning(f"BM25 document chunks not found at {bm25_docs_path}")
            logger.info("Creating empty file for BM25")
            os.makedirs(os.path.dirname(bm25_docs_path), exist_ok=True)
            with open(bm25_docs_path, "w") as f:
                f.write("")  # Create empty file
        
        # Create RAG service
        rag_service = RAGService(
            vector_store=vector_store,
            llm_provider=llm_provider,
            use_hybrid_search=config.USE_HYBRID_SEARCH,
            use_reranker=config.RERANKER_ENABLED,
            check_hallucinations=config.HALLUCINATION_CHECK_ENABLED,
            confidence_threshold=0.6,
            vector_weight=0.7,
            bm25_weight=0.3,
            retrieval_k=config.RETRIEVAL_K,
            bm25_docs_path=bm25_docs_path
        )
        
        return rag_service
    except Exception as e:
        logger.error(f"Error initializing RAG service: {str(e)}")
        st.error(f"Error initializing RAG service: {str(e)}")
        st.stop()

def main():
    """Main application entry point."""
    try:
        # Output some diagnostic information at startup
        logger.info(f"Starting PDF Chatbot with Python {sys.version}")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Load environment variables
        load_environment()
        
        # Initialize components
        metrics_manager = initialize_metrics()
        vector_store = initialize_vector_store()
        document_processor = initialize_document_processor()
        llm_provider = initialize_llm_provider()
        rag_service = initialize_rag_service(vector_store, llm_provider)
        
        # Initialize the chat UI
        chat_ui = ChatUI(rag_service, metrics_manager)
        
        # Render the chat UI
        chat_ui.render()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main() 