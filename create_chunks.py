#!/usr/bin/env python3

"""
Script to create document chunks for BM25 retrieval.
This extracts text from PDFs and creates a document_chunks.txt file.
"""

import os
import sys
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def main():
    parser = argparse.ArgumentParser(description='Create document chunks for BM25 retrieval')
    parser.add_argument('--pdf_path', type=str, default='data/random machine learing pdf.pdf', 
                        help='Path to the PDF file')
    parser.add_argument('--output_path', type=str, default='data/document_chunks.txt',
                        help='Path to output the document chunks')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Size of document chunks')
    parser.add_argument('--chunk_overlap', type=int, default=200,
                        help='Overlap between chunks')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found at {args.pdf_path}")
        sys.exit(1)
    
    print(f"Loading PDF from {args.pdf_path}...")
    loader = PyPDFLoader(args.pdf_path)
    pages = loader.load()
    
    print(f"Loaded {len(pages)} pages from PDF")
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Split pages into chunks
    all_text = ""
    for page in pages:
        all_text += page.page_content + "\n\n"
    
    chunks = text_splitter.split_text(all_text)
    
    print(f"Created {len(chunks)} chunks")
    
    # Write chunks to file
    with open(args.output_path, 'w') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i+1} ---\n")
            f.write(chunk)
            f.write("\n\n")
    
    print(f"Wrote {len(chunks)} chunks to {args.output_path}")
    print("Done!")

if __name__ == "__main__":
    main() 