from typing import List, Optional, Dict, Any
import os
import logging
import hashlib
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process documents for RAG."""
    
    def __init__(
        self, 
        chunk_size: int = config.CHUNK_SIZE, 
        chunk_overlap: int = config.CHUNK_OVERLAP
    ):
        """Initialize document processor."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Process a PDF document into text chunks.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List[Document]: Chunked document splits
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"PDF file not found at {file_path}")
                raise FileNotFoundError(f"PDF file not found at {file_path}")
                
            logger.info(f"Loading PDF from {file_path}...")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            logger.info(f"Loaded {len(docs)} pages, splitting text into chunks...")
            text_splitter = self._create_text_splitter()
            splits = text_splitter.split_documents(docs)
            logger.info(f"Created {len(splits)} text chunks")
            
            # Add file hash to metadata to identify document version
            file_hash = self._calculate_file_hash(file_path)
            for split in splits:
                if 'source' not in split.metadata:
                    split.metadata['source'] = file_path
                split.metadata['file_hash'] = file_hash
                
            return splits
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
            
    def _create_text_splitter(self):
        """Create a text splitter with the configured parameters."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate a hash for the file to track versions."""
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:10]
            return file_hash
        except Exception as e:
            logger.error(f"Error calculating file hash: {str(e)}")
            return "unknown_hash"
            
    def save_chunks_for_bm25(self, chunks: List[Document], output_path: str) -> bool:
        """
        Save document chunks to a file for BM25 retrieval.
        
        Args:
            chunks: List of document chunks
            output_path: Path to save the chunks
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write chunks to file
            with open(output_path, 'w') as f:
                for i, chunk in enumerate(chunks):
                    f.write(f"--- Chunk {i+1} ---\n")
                    f.write(chunk.page_content)
                    f.write("\n\n")
            
            logger.info(f"Wrote {len(chunks)} chunks to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving chunks for BM25: {str(e)}")
            return False
            
    def get_document_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata about a document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dict[str, Any]: Document metadata
        """
        try:
            if not os.path.exists(file_path):
                return {"error": "File not found"}
                
            # Load document to get page count
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Get file stats
            file_stats = os.stat(file_path)
            file_hash = self._calculate_file_hash(file_path)
            
            return {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_size": file_stats.st_size,
                "page_count": len(docs),
                "file_hash": file_hash,
                "last_modified": file_stats.st_mtime
            }
        except Exception as e:
            logger.error(f"Error getting document metadata: {str(e)}")
            return {"error": str(e)} 