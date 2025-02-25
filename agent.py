from openai import OpenAI
import json
import os
from pytao import Tao
from bmad_db import BmadDatabase
import readline

class BmadTaoAgent:
    """
    AI agent for answering questions about Bmad and executing Tao commands.
    Combines vector database retrieval with Tao command execution.
    """
    def __init__(self, model_name="gpt-4o", verbose=False, db_path="faiss_clean", lattice_path=None):
        """
        Initialize the Bmad/Tao Agent.
        
        Args:
            model_name (str): OpenAI model to use
            verbose (bool): Whether to print verbose information
            db_path (str): Path to the FAISS database
            lattice_path (str): Path to the Bmad lattice file for Tao
        """
        self.client = OpenAI()
        self.model_name = model_name
        self.verbose = verbose
        
        # Initialize the database
        self.db = BmadDatabase(db_path=db_path, verbose=verbose)
        
        # Initialize Tao if lattice path is provided
        self.tao = None
        if lattice_path:
            self.init_tao(lattice_path)
        
        # Define tools for the model
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_tao_command",
                    "description": "Executes a command using Tao simulation software and returns the result. Use 'help' for available commands.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to execute via Tao."
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_bmad_manual",
                    "description": "Queries the Bmad manual for information on a specific topic.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query about Bmad to look up in the documentation."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_bmad_lattice",
                    "description": "Writes a Bmad lattice file to disk.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path where the lattice file should be saved."
                            },
                            "content": {
                                "type": "string",
                                "description": "The content of the Bmad lattice file to be written."
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_bmad_lattice",
                    "description": "Reads a Bmad lattice file from disk.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the Bmad lattice file to read."
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            }
        ]
        
        # Initialize conversation history
        self.messages = [
            {
                "role": "system",
                "content": """You are an expert assistant for Bmad and Tao, specialized in particle accelerator physics and simulations.
You have four main capabilities:
1. You can answer questions about Bmad by searching the manual using query_bmad_manual.
2. You can execute Tao commands using execute_tao_command to interact with the simulation.
3. You can write Bmad lattice files to disk using write_bmad_lattice.
4. You can read Bmad lattice files from disk using read_bmad_lattice.

When responding:
- For conceptual questions about Bmad, use query_bmad_manual to find relevant information.
- For running simulations or analyzing results, use execute_tao_command to interact with Tao.
- For lattice file operations, use the read and write lattice functions.
- For lattice creation, first search the manual for relevant element types and syntax before writing.
- When creating a lattice from scratch, follow best practices for Bmad lattice files.
- For complex tasks, break them down into steps and execute each step with appropriate tool calls.
- When searching the manual, start with broad queries and then refine with more specific follow-up queries.
- When working with Tao, you might need to execute multiple commands in sequence to complete an analysis.
- Be precise and technically accurate in your explanations.
- When executing Tao commands, explain what the command does and interpret the results.

If you don't know the answer, say so. Don't make up information."""
            }
        ]
        
        print(f"BmadTaoAgent initialized with model: {model_name}")
        if self.tao:
            print("Tao initialized successfully")
    
    def init_tao(self, lattice_path, options="-noplot"):
        """
        Initialize the Tao instance with a lattice file.
        
        Args:
            lattice_path (str): Path to the Bmad lattice file
            options (str): Additional options for Tao initialization
        """
        try:
            init_command = f"-lat {lattice_path} {options}"
            self.tao = Tao(init_command)
            return True
        except Exception as e:
            print(f"Error initializing Tao: {str(e)}")
            return False
    
    def execute_tao_command(self, command):
        """
        Execute a command in Tao and return the result.
        
        Args:
            command (str): The command to execute
            
        Returns:
            list: Lines of output from the command
        """
        if not self.tao:
            return ["Error: Tao is not initialized. Please initialize with a lattice file."]
        
        if self.verbose:
            print(f"Executing Tao command: {command}")
            
        try:
            result = self.tao.cmd(command)
            return result
        except Exception as e:
            error_msg = f"Error executing Tao command: {str(e)}"
            if self.verbose:
                print(error_msg)
            return [error_msg]
    
    def query_bmad_manual(self, query):
        """
        Query the Bmad manual for information.
        
        Args:
            query (str): The query to look up
            
        Returns:
            str: Relevant information from the manual
        """
        if self.verbose:
            print(f"Querying Bmad manual: {query}")
            
        context = self.db.get_context(query)
        return context
        
    def write_bmad_lattice(self, file_path, content):
        """
        Write a Bmad lattice file to disk after user confirmation.
        
        Args:
            file_path (str): Path where to save the file
            content (str): Content of the Bmad lattice file
            
        Returns:
            str: Success or error message
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Show content and get user confirmation
            print("\n=== PREVIEW OF FILE TO BE WRITTEN ===")
            print(f"File: {file_path}")
            print("Content:")
            print("-" * 60)
            print(content)
            print("-" * 60)
            
            # Ask for confirmation
            confirmation = input("Do you want to write this file? (y/n): ").lower()
            if confirmation != 'y':
                return "File write cancelled by user."
            
            # Write the file if confirmed
            with open(file_path, 'w') as f:
                f.write(content)
                
            return f"Successfully wrote Bmad lattice file to {file_path}"
        except Exception as e:
            error_msg = f"Error writing Bmad lattice file: {str(e)}"
            if self.verbose:
                print(error_msg)
            return error_msg
            
    def read_bmad_lattice(self, file_path):
        """
        Read a Bmad lattice file from disk.
        
        Args:
            file_path (str): Path to the Bmad lattice file
            
        Returns:
            str: Content of the Bmad lattice file or error message
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            return content
        except Exception as e:
            error_msg = f"Error reading Bmad lattice file: {str(e)}"
            if self.verbose:
                print(error_msg)
            return error_msg
    
    def process_message(self, user_input):
        """
        Process a user message and generate a response.
        
        Args:
            user_input (str): The user's message
            
        Returns:
            str: The assistant's response
        """
        # Add user message to conversation history
        self.messages.append({"role": "user", "content": user_input})
        
        # Keep processing until no more tool calls
        while True:
            # Prepare messages based on model
            messages_to_send = self.messages.copy()

            # Get model completion with tool calling
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_to_send,
                tools=self.tools,
                tool_choice="auto"
            )
            
            # Add the response to the conversation history
            assistant_message = completion.choices[0].message
            self.messages.append(assistant_message)
            
            # Process any tool calls
            tool_calls = assistant_message.tool_calls
            if not tool_calls:
                # No more tool calls, we're done
                break
                
            # Process all tool calls
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # Execute the appropriate tool
                if function_name == "execute_tao_command":
                    result = self.execute_tao_command(arguments['command'])
                    result_content = '\n'.join(result)
                    if self.verbose:
                        print(f"Tool result for {function_name}: {result_content[:100]}...")
                elif function_name == "query_bmad_manual":
                    result_content = self.query_bmad_manual(arguments['query'])
                    if self.verbose:
                        print(f"Tool result for {function_name}: {arguments['query']}, length: {len(result_content)}")
                elif function_name == "write_bmad_lattice":
                    result_content = self.write_bmad_lattice(arguments['file_path'], arguments['content'])
                    if self.verbose:
                        print(f"Tool result for {function_name}: {result_content}")
                elif function_name == "read_bmad_lattice":
                    result_content = self.read_bmad_lattice(arguments['file_path'])
                    if self.verbose:
                        print(f"Tool result for {function_name}: file length: {len(result_content)}")
                else:
                    result_content = f"Error: Unknown tool {function_name}"
                    if self.verbose:
                        print(f"Error: Unknown tool {function_name}")
                
                # Add the tool response to the conversation
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_content
                })
        
        # Return the final response content
        return completion.choices[0].message.content
    
    def clear_history(self):
        """Clear the conversation history except for the system message"""
        system_message = self.messages[0]  # Save the system message
        self.messages = [system_message]   # Reset history with just system message
        print("Conversation history cleared.")
        
    def show_help(self):
        """Show available commands"""
        help_text = """
Available commands:
  /help   - Show this help message
  /clear  - Clear conversation history
  /exit   - Exit the program
"""
        print(help_text)
        
    def run_interactive(self):
        """Run the agent in interactive mode"""
        # Define command handlers
        commands = {
            "/clear": self.clear_history,
            "/exit": lambda: "exit",  # Special return value to signal exit
            "/help": self.show_help
        }
        
        print("Type /help to see available commands")
        
        while True:
            user_input = input("\nAsk about Bmad/Tao: ")
            
            # Check if input is a command
            if user_input.startswith('/'):
                command = user_input.split()[0]  # Get first word (the command)
                if command in commands:
                    result = commands[command]()
                    # Special case for exit command
                    if result == "exit":
                        print("Goodbye!")
                        break
                    continue
                else:
                    print(f"Unknown command: {command}")
                    print("Type /help to see available commands")
                    continue
            
            # Process regular user input
            response = self.process_message(user_input)
            print("\n" + response)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bmad/Tao Assistant")
    parser.add_argument("--model", type=str, default="gpt-4o", 
                        help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument("--verbose", action="store_true", 
                        help="Print verbose information")
    parser.add_argument("--db-path", type=str, default="faiss_clean",
                        help="Path to the FAISS database (default: faiss_clean)")
    parser.add_argument("--lattice", type=str, 
                        help="Path to the Bmad lattice file for Tao")
    parser.add_argument("query", nargs="?", type=str, 
                        help="Optional query (if not provided, runs in interactive mode)")
    
    args = parser.parse_args()
    
    agent = BmadTaoAgent(
        model_name=args.model, 
        verbose=args.verbose,
        db_path=args.db_path,
        lattice_path=args.lattice
    )
    
    if args.query:
        # Single query mode
        response = agent.process_message(args.query)
        print(response)
    else:
        # Interactive mode
        agent.run_interactive()
