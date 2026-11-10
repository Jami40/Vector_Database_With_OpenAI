# Vector Databases for GenAI

> Three practical Jupyter Notebook examples for building semantic-search and Retrieval-Augmented Generation (RAG) workflows with ChromaDB, FAISS, and Pinecone.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-embeddings-412991?logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-local%20vector%20store-FF6B35)
![FAISS](https://img.shields.io/badge/FAISS-similarity%20search-0467DF)
![Pinecone](https://img.shields.io/badge/Pinecone-cloud%20vector%20database-00A98F)

## Overview

This project compares three common approaches to vector search:

| Notebook | Vector database | Embedding approach | Main use case |
|---|---|---|---|
| [`ChoromaDB_Vector_Database.ipynb`](./ChoromaDB_Vector_Database.ipynb) | ChromaDB | OpenAI embeddings | Local RAG with LangChain |
| [`FAISS_Database.ipynb`](./FAISS_Database.ipynb) | FAISS | OpenAI embeddings | Lightweight in-memory similarity search |
| [`Pinecone_Cloud_Database.ipynb`](./Pinecone_Cloud_Database.ipynb) | Pinecone | Pinecone integrated embedding | Managed cloud vector search |

The notebooks are independent learning examples. Run the one that matches the storage, deployment, and retrieval model you want to explore.

## Architecture

The project’s reference architecture illustrates the core RAG lifecycle:

![RAG architecture](<./images/Screenshot%202026-11-09%20162243.png>)

### Common retrieval lifecycle

```text
Raw documents
      |
      v
Extract and prepare text
      |
      v
Create chunks or records
      |
      v
Generate embeddings
      |
      +--------------------+--------------------+
      |                    |                    |
      v                    v                    v
   ChromaDB              FAISS              Pinecone
   Local store         In-memory index      Cloud index
      |                    |                    |
      +--------------------+--------------------+
                           |
                           v
                 Retrieve relevant context
                           |
                           v
                    Answer or inspect
```

The included architecture image focuses on the ChromaDB RAG flow: documents are extracted, chunked, embedded, stored in a vector database, retrieved for a question, and passed to an LLM.

## Notebook implementations

### 1. ChromaDB + LangChain RAG

[`ChoromaDB_Vector_Database.ipynb`](./ChoromaDB_Vector_Database.ipynb) demonstrates a complete local RAG pipeline:

1. Download and extract a sample article archive.
2. Load `.txt` files with `DirectoryLoader` and `TextLoader`.
3. Split documents with `RecursiveCharacterTextSplitter`.
4. Use 1,000-character chunks with 200-character overlap.
5. Generate OpenAI embeddings.
6. Persist vectors locally in the `db/` directory.
7. Retrieve relevant chunks with a LangChain retriever.
8. Use `RetrievalQA` to generate an answer and return source documents.

The notebook demonstrates questions such as:

```python
query = "How much money did Microsoft raise?"
query = "What is the news about Pando?"
```

### 2. FAISS + OpenAI embeddings

[`FAISS_Database.ipynb`](./FAISS_Database.ipynb) demonstrates a small, fast, in-memory similarity-search index:

1. Define a collection of example texts.
2. Generate embeddings with OpenAI’s `text-embedding-ada-002` model.
3. Normalize vectors for cosine similarity.
4. Create a `faiss.IndexFlatIP` index.
5. Add the vectors to the index.
6. Embed a query and retrieve the top two matches.
7. Use `text_id_map` to resolve vector IDs back to original text.

Because normalized vectors are searched with inner product, `IndexFlatIP` behaves as a cosine-similarity index in this example.

### 3. Pinecone Cloud with integrated embeddings

[`Pinecone_Cloud_Database.ipynb`](./Pinecone_Cloud_Database.ipynb) demonstrates managed cloud vector search:

1. Install the Pinecone Python SDK.
2. Initialize a Pinecone client with `PINECONE_API_KEY`.
3. Create a dense index named `dense-index` if it does not already exist.
4. Configure Pinecone’s integrated `llama-text-embed-v2` embedding model.
5. Upsert text records and metadata into the `example-namespace` namespace.
6. Inspect index statistics.
7. Search using a natural-language query such as `Famous historical structures and monuments`.
8. Delete the index when cleanup is required.

Pinecone handles the embedding and vector indexing through its cloud service, so this notebook does not manually create local embedding arrays.

## Features

- Compare local, in-memory, and managed cloud vector databases.
- Perform semantic search using natural-language queries.
- Generate OpenAI embeddings for ChromaDB and FAISS.
- Use Pinecone integrated embeddings for cloud indexing.
- Persist a local ChromaDB collection.
- Retrieve top-k similar results with FAISS.
- Store records and metadata in a Pinecone namespace.
- Explore RAG answers with source documents.
- Inspect index statistics and clean up resources.

## Project structure

```text
.
├── ChoromaDB_Vector_Database.ipynb  # Local ChromaDB + LangChain RAG
├── FAISS_Database.ipynb             # In-memory FAISS similarity search
├── Pinecone_Cloud_Database.ipynb    # Managed Pinecone cloud search
├── Vector_Database_GENAI.pdf        # Project/reference material
├── images/
│   └── Screenshot 2026-11-09 162243.png  # Reference architecture
└── README.md
```

Generated resources are created only when notebooks run:

- `new_articles/` and `new_articles.zip` for the ChromaDB example
- `db/` for the persisted ChromaDB collection
- A remote `dense-index` and namespace for the Pinecone example

## Requirements

- Python 3.10 or newer
- Jupyter Notebook, JupyterLab, or Google Colab
- Internet access
- An OpenAI API key for the ChromaDB and FAISS notebooks
- A Pinecone API key for the Pinecone notebook
- A compatible `faiss-cpu` wheel for the local environment

## Installation

### ChromaDB notebook

```bash
pip install chromadb openai langchain langchain-community tiktoken
```

### FAISS notebook

The notebook uses the legacy OpenAI Python client API:

```bash
pip install faiss-cpu numpy openai==0.28
```

### Pinecone notebook

```bash
pip install pinecone
```

Run the installation cells at the beginning of the notebook you want to use. Pin versions in a dependency file before using these examples in a production application.

## Configuration and secrets

Never put real API keys directly in a notebook or commit them to Git. Configure credentials through environment variables.

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:PINECONE_API_KEY = "your-pinecone-api-key"
```

### macOS/Linux

```bash
export OPENAI_API_KEY="your-openai-api-key"
export PINECONE_API_KEY="your-pinecone-api-key"
```

The OpenAI notebooks read the key like this:

```python
import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]
```

The Pinecone notebook reads its key like this:

```python
import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
```

If a key has ever been committed, revoke it immediately, remove it from the commit history, and create a replacement key.

## Running the notebooks

1. Clone or download this repository.
2. Open one of the three notebooks.
3. Install that notebook’s dependencies.
4. Set the required environment variables.
5. Run the cells from top to bottom.
6. Modify the sample query or records to experiment with retrieval.

### ChromaDB query

The ChromaDB notebook creates a `qa_chain` and returns an answer with source documents.

### FAISS query

Change the query and top-k value:

```python
query = "What is FAISS?"
k = 2
distances, indices = index.search(query_vector.reshape(1, -1), k)
```

### Pinecone query

Search the dense index through the configured namespace:

```python
query = "Famous historical structures and monuments"

results = dense_index.search(
    namespace="example-namespace",
    query={"inputs": {"text": query}, "top_k": 5},
)
```

## Local and cloud cleanup

The ChromaDB notebook includes cleanup logic for its local collection:

```python
vectordb.delete_collection()
vectordb.persist()
```

The Pinecone notebook includes cleanup logic for the remote index:

```python
pc.delete_index(index_name)
```

Run Pinecone cleanup only when the cloud index is no longer needed. Deleting it removes the remote index and its stored records.

## Limitations and future improvements

These notebooks are educational examples rather than production services. Before production use, consider:

- Pinning all dependencies and documenting tested Python versions.
- Migrating legacy LangChain and OpenAI imports to current APIs.
- Adding automated tests for ingestion, retrieval, and answer quality.
- Adding retries, validation, logging, and explicit API error handling.
- Evaluating retrieval quality with a representative question-and-answer dataset.
- Adding metadata filters, deduplication, and incremental indexing.
- Separating secrets from notebooks with a secret manager.
- Adding a web or API interface.
- Applying authentication, authorization, rate limits, and cost controls.
- Monitoring Pinecone index usage and OpenAI embedding costs.

## License

No license file is currently included. Add a license before distributing or publishing this project.
