#!/bin/bash
set -e

echo "Starting PDF Chatbot initialization..."

# Check required environment variables
echo "Checking environment variables..."
if [ -z "$OPENAI_API_KEY" ]; then
  echo "ERROR: OPENAI_API_KEY environment variable is not set."
  echo "Please set this variable before starting the application."
  exit 1
fi

if [ -z "$PINECONE_API_KEY" ]; then
  echo "ERROR: PINECONE_API_KEY environment variable is not set."
  echo "Please set this variable before starting the application."
  exit 1
fi

if [ -z "$PINECONE_ENVIRONMENT" ]; then
  echo "PINECONE_ENVIRONMENT not set, defaulting to us-east-1..."
  export PINECONE_ENVIRONMENT="us-east-1"
fi

if [ -z "$PINECONE_INDEX_NAME" ]; then
  echo "ERROR: PINECONE_INDEX_NAME environment variable is not set."
  echo "Please set this variable before starting the application."
  exit 1
fi
echo "Using Pinecone index: $PINECONE_INDEX_NAME"

# Add current directory to Python path
export PYTHONPATH=${PYTHONPATH}:$(pwd)
echo "Setting PYTHONPATH to: ${PYTHONPATH}"

# Required ports
STREAMLIT_PORT=8501
METRICS_PORT=${METRICS_PORT:-8099}

# Force kill any processes using our ports
echo "Forcefully clearing any processes using ports $STREAMLIT_PORT and $METRICS_PORT..."

# For Linux/macOS
if command -v lsof &> /dev/null; then
  lsof -ti:$STREAMLIT_PORT | xargs kill -9 2>/dev/null || true
  lsof -ti:$METRICS_PORT | xargs kill -9 2>/dev/null || true
# For other systems or fallback
elif command -v netstat &> /dev/null; then
  # Get PIDs using netstat and kill them
  PIDs=$(netstat -tulnp 2>/dev/null | grep ":$STREAMLIT_PORT" | awk '{print $7}' | cut -d'/' -f1)
  for PID in $PIDs; do
    kill -9 $PID 2>/dev/null || true
  done
  
  PIDs=$(netstat -tulnp 2>/dev/null | grep ":$METRICS_PORT" | awk '{print $7}' | cut -d'/' -f1)
  for PID in $PIDs; do
    kill -9 $PID 2>/dev/null || true
  done
fi

# Wait to ensure ports are fully released
echo "Waiting for ports to be released..."
sleep 3

# Initialize vector store if needed
echo "Checking if vector store index exists..."
if [ ! -f "vectorstore/index_name.txt" ] && [ -z "$PINECONE_INDEX_NAME" ]; then
  echo "Vector store index not found and no PINECONE_INDEX_NAME provided. Initializing vector store..."
  # Use data module for vector store initialization
  python -c "
import os
import sys
import time

# Add the current directory to the path
sys.path.insert(0, os.getcwd())

from data.document import DocumentProcessor
from data.vector_store import PineconeVectorStoreWrapper
from langchain_openai import OpenAIEmbeddings

# Initialize vector store
pdf_path = os.environ.get('PDF_PATH', '/app/data/random_machine_learning_pdf.pdf')
print(f'Initializing vector store from PDF: {pdf_path}')

# Create document processor
processor = DocumentProcessor()
docs = processor.process_pdf(pdf_path)

# Get embeddings service
embeddings = OpenAIEmbeddings(api_key=os.environ.get('OPENAI_API_KEY'))

# Use environment variable if provided, otherwise create unique name
env_index_name = os.environ.get('PINECONE_INDEX_NAME')
if env_index_name:
    index_name = env_index_name
    print(f'Using Pinecone index from environment: {index_name}')
else:
    # Create a unique index name
    index_name = f'pdf-chatbot-{int(time.time())}'
    print(f'Creating Pinecone index: {index_name}')

with open('vectorstore/index_name.txt', 'w') as f:
    f.write(index_name)

# Create vector store and add documents
vector_store = PineconeVectorStoreWrapper.from_documents(
    documents=docs,
    embeddings=embeddings,
    index_name=index_name
)

# Save chunks for BM25
processor.save_chunks_for_bm25(docs, 'data/document_chunks.txt')

# Verify vector store initialization
print('Vector store initialization complete')
"
  # Check if initialization was successful
  if [ ! -f "vectorstore/index_name.txt" ]; then
    echo "Vector store initialization failed!"
    echo "Check environment variables: OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT"
    exit 1
  else
    echo "Vector store initialized successfully."
  fi
else
  echo "Vector store index already exists."
fi

# Start explicit metrics server using monitoring module if available
echo "Starting metrics server on port $METRICS_PORT..."
if [ -d "monitoring" ]; then
  python -c "
import os
import sys

# Add the current directory to the path
sys.path.insert(0, os.getcwd())

from monitoring.metrics import MetricsManager

# Get port from environment
port = int(os.environ.get('METRICS_PORT', 8099))

# Initialize metrics manager
metrics = MetricsManager(metrics_port=port, enable_metrics=True)
metrics.start_metrics_server()
print(f'Metrics server started on port {port}')
" &
else
  python -c "
import os
import sys
import time

# Add the current directory to the path
sys.path.insert(0, os.getcwd())

from prometheus_client import start_http_server, Counter

# Define some basic metrics
requests = Counter('chatbot_requests_total', 'Total number of requests')
requests.inc(0)  # Initialize with 0

# Start server on the metrics port, binding to all interfaces
start_http_server(${METRICS_PORT}, addr='0.0.0.0')
print(f'Metrics server started on port ${METRICS_PORT}')

# Keep server running in background
" &
fi

METRICS_PID=$!
sleep 2

# Check if metrics server started successfully
if ps -p $METRICS_PID > /dev/null; then
  echo "Metrics server started successfully (PID: $METRICS_PID)"
else
  echo "WARNING: Failed to start metrics server. Continuing anyway..."
fi

# Set metrics port for Streamlit app
export METRICS_PORT=${METRICS_PORT}

# Verify Python and Streamlit are available
if ! command -v python &> /dev/null; then
  echo "ERROR: Python is not installed or not in PATH"
  exit 1
fi

if ! python -c "import streamlit" &> /dev/null; then
  echo "ERROR: Streamlit is not installed. Please install it with: pip install streamlit"
  exit 1
fi

# Start Streamlit with error handling
echo "Starting Streamlit server on port $STREAMLIT_PORT..."
echo "Metrics will be available on port $METRICS_PORT"
if ! python -m streamlit run streamlit_app.py; then
  echo "ERROR: Failed to start Streamlit application!"
  kill $METRICS_PID 2>/dev/null || true
  exit 1
fi 