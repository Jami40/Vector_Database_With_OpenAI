# Vector Databases for GenAI

> Practical notebook examples for building semantic search systems with ChromaDB and FAISS using OpenAI embeddings.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=chainlink&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-FF6B35)
![FAISS](https://img.shields.io/badge/FAISS-similarity%20search-0467DF)
![OpenAI](https://img.shields.io/badge/OpenAI-embeddings%20%2B%20LLM-412991?logo=openai&logoColor=white)

## Overview

This project contains two complementary vector-search implementations:

- **ChromaDB + LangChain:** an end-to-end Retrieval-Augmented Generation (RAG) workflow that loads articles, persists embeddings, retrieves relevant context, and generates answers with source documents.
- **FAISS + OpenAI:** a focused semantic-search example that embeds text, normalizes vectors for cosine similarity, and searches a FAISS index.

See [`ChoromaDB_Vector_Database.ipynb`](./ChoromaDB_Vector_Database.ipynb) and [`FAISS_Database.ipynb`](./FAISS_Database.ipynb) for the complete examples.

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

## Implementations

### ChromaDB RAG pipeline

The ChromaDB notebook loads `.txt` articles, splits them into 1,000-character chunks with 200-character overlap, creates OpenAI embeddings, and persists them in a local `db/` directory. A LangChain retriever then supplies relevant chunks to a `RetrievalQA` chain, which returns an answer and its source documents.

### FAISS semantic search

The FAISS notebook demonstrates a lightweight in-memory vector index:

1. Define example text documents.
2. Generate OpenAI embeddings with `text-embedding-ada-002`.
3. Normalize vectors for cosine similarity.
4. Add vectors to a `faiss.IndexFlatIP` index.
5. Embed a query and retrieve the top two matching texts.

FAISS is used here as the similarity-search engine, while the original text is kept in an `text_id_map` for lookup after searching.

## Features

- **Semantic search** over document content rather than keyword-only matching.
- **Persistent vector storage** using ChromaDB and an in-memory FAISS alternative.
- **Configurable retrieval**, including a demonstrated `k=2` result limit in both examples.
- **Source-aware RAG answers** through `return_source_documents=True`.
- **Cosine-similarity search** using normalized embeddings and FAISS inner product.
- **Notebook-first learning experience** that makes each RAG step inspectable.
- **Reusable document ingestion and embedding patterns** for additional content.

## Project structure

```text
.
├── ChoromaDB_Vector_Database.ipynb  # End-to-end RAG implementation
├── FAISS_Database.ipynb             # FAISS semantic-search implementation
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
- A C++ compiler/runtime compatible with the selected `faiss-cpu` wheel, if installing FAISS locally
- Internet access for package installation, sample-data download, embeddings, and LLM requests

## Installation

Install the libraries used by the notebook:

```bash
pip install chromadb openai langchain langchain-community tiktoken faiss-cpu numpy
```

The FAISS notebook uses the legacy OpenAI Python client, so its first cells install:

```bash
pip install faiss-cpu numpy openai==0.28
```

If you are using Google Colab, run the installation cells at the beginning of the notebook you want to use.

## Configuration

Set the API key in the environment before creating embeddings or querying the model:

```python
import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]
```

For local development, prefer an environment variable instead of placing a real key in a notebook:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "your-api-key"

# macOS/Linux
export OPENAI_API_KEY="your-api-key"
```

Never commit API keys, generated credentials, or the ChromaDB persistence directory to source control.

## Running the notebooks

1. Open either [`ChoromaDB_Vector_Database.ipynb`](./ChoromaDB_Vector_Database.ipynb) or [`FAISS_Database.ipynb`](./FAISS_Database.ipynb).
2. Install the dependencies.
3. Configure `OPENAI_API_KEY`.
4. Run the cells from top to bottom.

For the ChromaDB notebook, ask questions using the `qa_chain` examples near the end of the notebook. For the FAISS notebook, change the `query` value and run the search cell to retrieve the most similar texts.

The notebook currently uses these example questions:

```python
query = "How much money did Microsoft raise?"
query = "What is the news about Pando?"
```

The response includes the generated answer and the paths of the source documents used to produce it.

The FAISS example returns matching text and similarity scores:

```python
query = "What is FAISS?"
k = 2
distances, indices = index.search(query_vector.reshape(1, -1), k)
```

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

These are educational notebook implementations, not production services. Potential improvements include:

- Pinning dependency versions in a `requirements.txt` or `pyproject.toml`.
- Migrating deprecated LangChain imports to the current package layout.
- Adding a web or API interface.
- Adding evaluation datasets for retrieval and answer quality.
- Adding metadata filters, document deduplication, and incremental indexing.
- Moving secrets and vector storage into managed infrastructure.
- Adding automated tests and error handling for failed downloads or API requests.
- Replacing the legacy `openai==0.28` API usage with the current OpenAI client.

## License

No license file is currently included. Add a license before distributing or publishing the project.
