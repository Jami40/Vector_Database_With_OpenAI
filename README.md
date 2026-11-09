# ChromaDB Vector Database for GenAI

> A practical Retrieval-Augmented Generation (RAG) notebook for searching a document collection with semantic similarity and answering questions with source citations.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=chainlink&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-FF6B35)
![OpenAI](https://img.shields.io/badge/OpenAI-embeddings%20%2B%20LLM-412991?logo=openai&logoColor=white)

## Overview

This project demonstrates how to build a searchable knowledge base from text documents:

1. Load raw article files.
2. Split documents into overlapping chunks.
3. Generate vector embeddings with OpenAI.
4. Persist the embeddings in a local ChromaDB database.
5. Retrieve the most relevant chunks for a question.
6. Pass the retrieved context to an OpenAI language model.
7. Return an answer together with the source documents used.

The complete workflow is available in [`ChoromaDB_Vector_Database.ipynb`](./ChoromaDB_Vector_Database.ipynb).

## Architecture

The following diagram represents the implemented RAG pipeline:

![RAG architecture](<./images/Screenshot%202026-11-09%20162243.png>)

### Request flow

```text
Text documents
      |
      v
DirectoryLoader + TextLoader
      |
      v
RecursiveCharacterTextSplitter
(chunk_size=1000, chunk_overlap=200)
      |
      v
OpenAIEmbeddings
      |
      v
ChromaDB (persisted locally in ./db)
      |
      v
Retriever (top-k relevant chunks)
      |
      v
RetrievalQA + OpenAI LLM
      |
      v
Answer + source document paths
```

## Features

- **Semantic search** over document content rather than keyword-only matching.
- **Persistent vector storage** using ChromaDB.
- **Configurable retrieval**, including a demonstrated `k=2` result limit.
- **Source-aware answers** through `return_source_documents=True`.
- **Notebook-first learning experience** that makes each RAG step inspectable.
- **Reusable document ingestion pattern** for additional `.txt` files.

## Project structure

```text
.
├── ChoromaDB_Vector_Database.ipynb  # End-to-end RAG implementation
├── Vector_Database_GENAI.pdf        # Project/reference material
├── images/
│   └── Screenshot 2026-11-09 162243.png  # Architecture diagram
└── README.md
```

> The notebook downloads a sample article archive into `new_articles/` and creates the local `db/` directory when its cells are executed. These generated directories are not part of the repository snapshot.

## Requirements

- Python 3.10 or newer
- Jupyter Notebook, JupyterLab, or Google Colab
- An OpenAI API key
- Internet access for package installation, sample-data download, embeddings, and LLM requests

## Installation

Install the libraries used by the notebook:

```bash
pip install chromadb openai langchain langchain-community tiktoken
```

If you are using Google Colab, run the installation cells at the beginning of the notebook.

## Configuration

Set the API key in the notebook before creating embeddings or querying the model:

```python
import os

os.environ["OPENAI_API_KEY"] = "your-api-key"
```

For local development, prefer an environment variable instead of placing a real key in a notebook:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "your-api-key"

# macOS/Linux
export OPENAI_API_KEY="your-api-key"
```

Never commit API keys, generated credentials, or the ChromaDB persistence directory to source control.

## Running the notebook

1. Open [`ChoromaDB_Vector_Database.ipynb`](./ChoromaDB_Vector_Database.ipynb).
2. Install the dependencies.
3. Configure `OPENAI_API_KEY`.
4. Run the cells from top to bottom.
5. Ask questions using the `qa_chain` examples near the end of the notebook.

The notebook currently uses these example questions:

```python
query = "How much money did Microsoft raise?"
query = "What is the news about Pando?"
```

The response includes the generated answer and the paths of the source documents used to produce it.

## Core implementation

### 1. Load and split documents

```python
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = DirectoryLoader(
    "/content/new_articles/",
    glob="./*.txt",
    loader_cls=TextLoader,
)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(documents)
```

### 2. Create and persist the vector database

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="db",
)
vectordb.persist()
```

### 3. Retrieve context and answer questions

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

vectordb = Chroma(
    persist_directory="db",
    embedding_function=embedding,
)

retriever = vectordb.as_retriever(search_kwargs={"k": 2})

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)
```

## Data lifecycle

The notebook includes a cleanup step that deletes the local ChromaDB collection and removes the `db/` directory. Run it only when you want to rebuild the index:

```python
vectordb.delete_collection()
vectordb.persist()
```

## Limitations and next steps

This is an educational notebook implementation, not a production service. Potential improvements include:

- Pinning dependency versions in a `requirements.txt` or `pyproject.toml`.
- Migrating deprecated LangChain imports to the current package layout.
- Adding a web or API interface.
- Adding evaluation datasets for retrieval and answer quality.
- Adding metadata filters, document deduplication, and incremental indexing.
- Moving secrets and vector storage into managed infrastructure.
- Adding automated tests and error handling for failed downloads or API requests.

## License

No license file is currently included. Add a license before distributing or publishing the project.
