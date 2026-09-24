
# Enterprise RAG Document Intelligence Assistant

A production-ready Retrieval-Augmented Generation (RAG) pipeline designed for secure, context-aware enterprise document querying. This assistant processes local text documents, generates vector embeddings, retrieves highly relevant context, and maintains strict audit logs of all query executions.

## Architecture & Tech Stack
* **Orchestration:** LangChain
* **Embeddings:** Hugging Face (`all-MiniLM-L6-v2`)
* **Vector Database:** ChromaDB
* **Audit & Performance Logging:** SQLite
* **Language:** Python

## Key Features
* **Robust Data Ingestion:** Automated text extraction and optimal chunking with built-in error handling.
* **Privacy-First Vector Retrieval:** Generates embeddings and executes semantic searches entirely locally using open-source Hugging Face models and ChromaDB.
* **Enterprise Telemetry:** Automated SQLite logging captures query timestamps, input payloads, exact response texts, character counts, and retrieval metrics for continuous performance auditing.

## Project Structure
ingest.py: Handles document loading, data cleaning, and text chunking.

vector_store.py: Manages ChromaDB initialization and HuggingFace embedding integration.

main.py: The core execution engine that wires together the retriever, prompt templates, LLM generation, and SQLite database commits.

requirements.txt: Strict versioning for all project dependencies.
