#!/usr/bin/env python3
"""
Test script for QdrantDatabase
"""
import os
import sys
from qdrant_db import QdrantDatabase

def main():
    """Main test function for Qdrant database"""
    # Create database instance with verbose output
    db = QdrantDatabase(
        collection_name="test_collection",
        path="./test_qdrant_data",
        verbose=True
    )
    
    print("Testing Qdrant Database...")
    
    # Sample documents
    documents = [
        {
            "text": "Beam tracking is a method used in accelerator physics to monitor and control particle beams.",
            "metadata": {"category": "tracking", "document_id": "doc1"}
        },
        {
            "text": "In Bmad, element orientation can be specified using three angles: Tilt, X_Pitch, and Y_Pitch.",
            "metadata": {"category": "orientation", "document_id": "doc2"}
        },
        {
            "text": "A quadrupole magnet is used to focus particle beams in accelerators.",
            "metadata": {"category": "elements", "document_id": "doc3"}
        },
        {
            "text": "Lattice files in Bmad use a specific syntax to define accelerator elements and their properties.",
            "metadata": {"category": "lattice", "document_id": "doc4"}
        },
        {
            "text": "Wakefield effects can significantly impact beam dynamics in high-intensity accelerators.",
            "metadata": {"category": "wakefields", "document_id": "doc5"}
        }
    ]
    
    # Add documents to the database
    print("\nAdding sample documents...")
    db.add_documents(documents)
    
    # Test basic search
    print("\nTesting basic search:")
    query = "particle beam tracking"
    print(f"Query: '{query}'")
    results = db.search(query, limit=3)
    for i, result in enumerate(results):
        print(f"Result {i+1} (Score: {result['score']:.4f}): {result['text']}")
    
    # Test filtered search
    print("\nTesting filtered search:")
    query = "Bmad"
    filter_category = db.create_filter("category", "lattice")
    print(f"Query: '{query}' with category='lattice'")
    results = db.search(query, limit=3, query_filter=filter_category)
    for i, result in enumerate(results):
        print(f"Result {i+1} (Score: {result['score']:.4f}): {result['text']}")
    
    # Test context retrieval
    print("\nTesting context retrieval:")
    query = "how do quadrupole magnets work?"
    print(f"Query: '{query}'")
    context = db.get_context(query, limit=2)
    print(context)
    
    print("\nTests completed successfully!")
    
    # Optional: Clean up
    choice = input("\nDelete test collection? (y/n): ").strip().lower()
    if choice == 'y':
        db.delete_collection()
        print("Test collection deleted.")

if __name__ == "__main__":
    main()