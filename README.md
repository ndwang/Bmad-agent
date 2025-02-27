# Bmad-Tao Agent

An intelligent assistant for working with the Bmad charged particle simulation library and Tao accelerator simulation software. This tool combines a RAG (Retrieval Augmented Generation) pipeline with direct Tao command execution capabilities to help users with both conceptual understanding and practical simulation tasks.

## Setup

1. Install dependencies:

```bash
pip install openai langchain_community langchain_huggingface faiss-cpu pytao
```

2. Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Interactive Mode

There are two interactive modes available:

1. **Agent Interactive Mode** - Full agent with Tao integration:
```bash
python agent.py
```

Special commands in agent mode:
- `/help` - Display available commands
- `/clear` - Reset conversation history
- `/exit` - Exit the program

2. **Database Query Mode** - Direct database queries without the agent:
```bash
python bmad_db.py
```

In database query mode, you can:
- Enter queries to search the Bmad documentation directly
- See raw context matches from the vector database
- Type 'quit' or 'exit' to end the session

### Single Query Mode

```bash
python agent.py "How does quadrupole tracking work in Bmad?"
```

### With Tao Integration

Initialize the agent with a lattice file to enable Tao simulation capabilities:

```bash
python agent.py --lattice path/to/your/lattice.bmad
```

### Additional Options

```bash
python agent.py --model gpt-4o --verbose --db-path faiss_clean "What is a group element in Bmad?"
```

## Features

- **Bmad Documentation Access**: Vector database built from the Bmad manual for accurate answers to technical questions
- **Tao Command Execution**: Direct integration with Tao simulation software
- **Lattice File Operations**: Read and write Bmad lattice files with intelligent assistance
- **Interactive Interface**: Command-line interface with conversation history
- **Semantic Search**: Retrieval of relevant context from the Bmad manual based on semantic similarity
- **Tool-based Architecture**: OpenAI function calling for specialized operations
- **Configurable Models**: Support for different OpenAI models with customizable parameters

## Capabilities

### Bmad Documentation Queries

The agent can provide detailed explanations about Bmad concepts, syntax, and functionality by searching the comprehensive Bmad manual:

- Element types and parameters
- Tracking methods and algorithms
- Lattice file syntax and best practices
- Physical models and simulation concepts

### Tao Command Execution

With an initialized lattice, the agent can:

- Run Tao commands and interpret results
- Perform beam physics calculations
- Generate and analyze simulation data
- Modify lattice parameters during runtime

### Lattice File Management

The agent assists with:

- Reading existing lattice files
- Creating new lattice files with proper syntax
- Modifying lattice elements and parameters
- Following best practices for Bmad lattice structure

## Project Structure

- `clean.py`: Preprocessing script for cleaning LaTeX files
- `build_db.ipynb`: Notebook for building the FAISS vector database
- `agent.py`: Main query interface combining RAG pipeline with Tao integration
- `bmad_db.py`: Vector database interface for Bmad documentation
- `bmad_doc/`: Original Bmad manual files
- `clean_bmad_doc/`: Processed text files
- `faiss_clean/`: Vector database files for semantic search

## Command Line Options

- `--model`: Specify which OpenAI model to use (default: gpt-4o)
- `--verbose`: Enable detailed logging information
- `--db-path`: Path to the FAISS database (default: faiss_clean)
- `--lattice`: Path to a Bmad lattice file to initialize Tao
- `query`: Optional query string (if omitted, runs in interactive mode)
