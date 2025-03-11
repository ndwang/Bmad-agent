#!/usr/bin/env python3
"""
Script to build Qdrant vector database from Bmad documentation
"""
import os
import glob
import argparse
from qdrant_db import QdrantDatabase

def load_docs_from_directory(directory_path):
    """
    Load documents from text files in a directory
    
    Args:
        directory_path: Path to directory containing text files
        
    Returns:
        List of document dicts with text and metadata
    """
    documents = []
    
    # Find all text files in the directory
    file_paths = glob.glob(os.path.join(directory_path, "*.txt"))
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract filename for metadata
            filename = os.path.basename(file_path)
            doc_name = os.path.splitext(filename)[0]
            
            # Add document with metadata
            documents.append({
                "text": content,
                "metadata": {
                    "source": "bmad_doc",
                    "filename": filename,
                    "doc_name": doc_name
                }
            })
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return documents

def main():
    """Main function to build the Qdrant database"""
    parser = argparse.ArgumentParser(description="Build Qdrant vector database from Bmad documentation")
    parser.add_argument("--docs-dir", default="clean_bmad_doc", help="Directory containing Bmad documentation text files")
    parser.add_argument("--tao-docs-dir", default="clean_tao_doc", help="Directory containing Tao documentation text files")
    parser.add_argument("--db-path", default="./qdrant_data", help="Path to store Qdrant database files")
    parser.add_argument("--collection", default="bmad_docs", help="Name of the Qdrant collection")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Initialize database
    db = QdrantDatabase(
        collection_name=args.collection,
        path=args.db_path,
        verbose=args.verbose
    )
    
    # Load Bmad documentation
    if os.path.isdir(args.docs_dir):
        print(f"Loading Bmad documentation from {args.docs_dir}...")
        bmad_docs = load_docs_from_directory(args.docs_dir)
        print(f"Loaded {len(bmad_docs)} Bmad documents")
        
        if bmad_docs:
            print("Adding Bmad documents to vector database...")
            db.add_documents(bmad_docs)
    else:
        print(f"Warning: Bmad docs directory '{args.docs_dir}' not found")
    
    # Load Tao documentation
    if os.path.isdir(args.tao_docs_dir):
        print(f"Loading Tao documentation from {args.tao_docs_dir}...")
        tao_docs = load_docs_from_directory(args.tao_docs_dir)
        print(f"Loaded {len(tao_docs)} Tao documents")
        
        if tao_docs:
            print("Adding Tao documents to vector database...")
            for doc in tao_docs:
                doc["metadata"]["source"] = "tao_doc"  # Mark as Tao docs
            db.add_documents(tao_docs)
    else:
        print(f"Warning: Tao docs directory '{args.tao_docs_dir}' not found")
    
    print("Database build complete.")

if __name__ == "__main__":
    main()