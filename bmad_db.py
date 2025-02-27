import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class BmadDatabase:
    """
    Handles vector database operations for the Bmad Agent system.
    Manages embeddings and retrieval from the FAISS vector database.
    """
    def __init__(self, db_path="faiss_clean", verbose=False):
        """
        Initialize the database with embeddings model and vector store.
        
        Args:
            db_path (str): Path to the FAISS database
            verbose (bool): Whether to print verbose retrieval information
        """
        self.verbose = verbose
        
        # Load embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load vector store
        self.vector_store = FAISS.load_local(
            db_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
    
    def get_context(self, query):
        """
        Retrieve relevant context from the vector store.
        
        Args:
            query (str): The user's query
            
        Returns:
            str: Formatted context from relevant documents
        """
        docs = self.retriever.invoke(query)
        
        # Format the retrieved documents into a context string
        context_parts = [f"Document {i+1}:\n{doc.page_content}\n" for i, doc in enumerate(docs)]
        context = "\n".join(context_parts)
        
        if self.verbose:
            print("\n=== Retrieved Context ===")
            print(context)
            print("=========================\n")
            
        return context
        
    def run_interactive(self):
        """
        Run an interactive query session with the database.
        Users can enter queries and see retrieved context.
        Enter 'quit' or 'exit' to end the session.
        """
        print("Welcome to Bmad Database Interactive Query")
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

if __name__ == "__main__":
    db = BmadDatabase(verbose=False)
    db.run_interactive()
