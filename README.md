# Bmad-agent

A RAG (Retrieval Augmented Generation) pipeline for answering questions about the Bmad charged particle simulation library using the Bmad manual as a knowledge source.

## Setup

1. Install dependencies:

```bash
pip install openai langchain_community langchain_huggingface faiss-cpu
```

2. Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Interactive Mode

Run the agent in interactive mode to ask questions about Bmad:

```bash
python agent.py
```

### Single Query Mode

```bash
python agent.py "How does quadrupole tracking work in Bmad?"
```

### Additional Options

```bash
python agent.py --model gpt-4o --verbose "What is a group element in Bmad?"
```

## Features

- Vector database built from the Bmad manual documentation
- Retrieval of relevant context based on semantic search
- Integration with OpenAI models for answering questions
- Interactive command-line interface

## Project Structure

- `clean.py`: Preprocessing script for cleaning LaTeX files
- `build_db.ipynb`: Notebook for building the FAISS vector database
- `agent.py`: Main query interface for the RAG pipeline
- `bmad_doc/`: Original Bmad manual files
- `clean_bmad_doc/`: Processed text files
- `faiss_tex/`: Vector database files