import os
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

class QdrantDatabase:
    """
    Handles vector database operations for the Bmad Agent system using Qdrant directly.
    Manages embeddings and retrieval from a local Qdrant vector database.
    """
    def __init__(
        self, 
        collection_name: str = "bmad_docs",
        path: str = "./qdrant_data",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_size: int = 384,
        verbose: bool = False
    ):
        """
        Initialize the database with embeddings model and local Qdrant client.
        
        Args:
            collection_name (str): Name of the Qdrant collection
            path (str): Path to the local directory for Qdrant storage
            embedding_model (str): Model to use for embeddings
            vector_size (int): Dimension of the embedding vectors
            verbose (bool): Whether to print verbose retrieval information
        """
        self.verbose = verbose
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Create the storage directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Load embeddings model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize Qdrant client with local persistence
        self.client = QdrantClient(path=path)
        
        # Ensure collection exists
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create the collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            if self.verbose:
                print(f"Created new collection: {self.collection_name}")

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        return self.embedding_model.encode(text).tolist()
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the Qdrant collection.
        
        Args:
            documents (List[Dict]): List of documents to add. 
                                   Each document should have at least 'text' and 'metadata' fields.
        """
        # Ensure collection exists
        self._ensure_collection_exists()
        
        points = []
        for i, doc in enumerate(documents):
            # Generate embedding for the document text
            vector = self._get_embedding(doc['text'])
            
            # Create point with ID, vector, and payload (metadata)
            point = PointStruct(
                id=len(points) + i,  # Generate unique ID
                vector=vector,
                payload={
                    "text": doc['text'],
                    **(doc.get('metadata', {}))
                }
            )
            points.append(point)
        
        # Upload all points at once
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        if self.verbose:
            print(f"Added {len(documents)} documents to collection {self.collection_name}")
    
    def search(self, query: str, limit: int = 5, query_filter: Optional[Filter] = None):
        """
        Search for documents similar to the query.
        
        Args:
            query (str): The search query
            limit (int): Maximum number of results to return
            query_filter (Filter): Optional filter to apply to the search
            
        Returns:
            List[Dict]: List of document dictionaries with scores
        """
        # Generate embedding for the query
        query_vector = self._get_embedding(query)
        
        # Search in the collection
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter
        )
        
        # Format results
        results = []
        for scored_point in search_result:
            results.append({
                'id': scored_point.id,
                'text': scored_point.payload.get('text', ''),
                'metadata': {k: v for k, v in scored_point.payload.items() if k != 'text'},
                'score': scored_point.score
            })
        
        return results
    
    def get_context(self, query: str, limit: int = 5, query_filter: Optional[Filter] = None):
        """
        Retrieve relevant context based on the query.
        
        Args:
            query (str): The user's query
            limit (int): Maximum number of results to return
            query_filter (Filter): Optional filter to apply to the search
            
        Returns:
            str: Formatted context from relevant documents
        """
        results = self.search(query, limit=limit, query_filter=query_filter)
        
        # Format the retrieved documents into a context string
        context_parts = [f"Document {i+1} (Score: {result['score']:.4f}):\n{result['text']}\n" 
                        for i, result in enumerate(results)]
        context = "\n".join(context_parts)
        
        if self.verbose:
            print("\n=== Retrieved Context ===")
            print(context)
            print("=========================\n")
            
        return context
    
    def create_filter(self, field: str, value: Any):
        """
        Create a filter condition for metadata field.
        
        Args:
            field (str): Metadata field to filter on
            value (Any): Value to match
            
        Returns:
            Filter: Qdrant filter object
        """
        return Filter(
            must=[
                FieldCondition(
                    key=field,
                    match=MatchValue(value=value)
                )
            ]
        )
        
    def run_interactive(self):
        """
        Run an interactive query session with the database.
        Users can enter queries and see retrieved context.
        Enter 'quit' or 'exit' to end the session.
        """
        print("Welcome to Bmad Qdrant Database Interactive Query")
        print("Enter your queries, or 'quit'/'exit' to end\n")
        
        while True:
            query = input("Query> ").strip()
            if query.lower() in ['quit', 'exit']:
                break
            if not query:
                continue
                
            context = self.get_context(query)
            print("\nRetrieved Context:")
            print("==================")
            print(context)
            print("==================\n")
            
    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(collection_name=self.collection_name)
        if self.verbose:
            print(f"Deleted collection: {self.collection_name}")

if __name__ == "__main__":
    db = QdrantDatabase(verbose=True)
    db.run_interactive()