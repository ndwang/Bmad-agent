import argparse
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
import textwrap

class BmadAgent:
    def __init__(self, model_name="gpt-4o", verbose=False):
        self.client = OpenAI()
        self.model_name = model_name
        self.verbose = verbose
        
        # Load embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load vector store
        self.vector_store = FAISS.load_local(
            "faiss_tex", 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        print(f"BmadAgent initialized with model: {model_name}")
        print("Type 'exit' to quit.")
    
    def get_context(self, query):
        """Retrieve relevant context from the vector store"""
        docs = self.retriever.invoke(query)
        
        # Format the retrieved documents into a context string
        context_parts = [f"Document {i+1}:\n{doc.page_content}\n" for i, doc in enumerate(docs)]
        context = "\n".join(context_parts)
        
        if self.verbose:
            print("\n=== Retrieved Context ===")
            #print(textwrap.shorten(context, width=200, placeholder="..."))
            print(context)
            print("=========================\n")
            
        return context
    
    def generate_response(self, query, context):
        """Generate a response using the retrieved context"""
        prompt = f"""You are an assistant with expertise in the Bmad charged particle simulation library.
Answer the user's question based on the context provided.
If you don't know the answer, say so. Don't make up information.

Context:
{context}

User Question: {query}
"""
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    def ask(self, query):
        """Process a user query and return a response"""
        context = self.get_context(query)
        response = self.generate_response(query, context)
        return response
    
    def run_interactive(self):
        """Run the agent in interactive mode"""
        while True:
            query = input("\nAsk about Bmad (or type 'exit' to quit): ")
            if query.lower() == 'exit':
                print("Goodbye!")
                break
                
            response = self.ask(query)
            print("\n" + response)

def main():
    parser = argparse.ArgumentParser(description="Bmad Document Assistant")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", 
                        help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--verbose", action="store_true", 
                        help="Print verbose retrieval information")
    parser.add_argument("query", nargs="?", type=str, 
                        help="Optional query (if not provided, runs in interactive mode)")
    
    args = parser.parse_args()
    
    agent = BmadAgent(model_name=args.model, verbose=args.verbose)
    
    if args.query:
        # Single query mode
        response = agent.ask(args.query)
        print(response)
    else:
        # Interactive mode
        agent.run_interactive()

if __name__ == "__main__":
    main()
