# LanceDB Chatbot Pipeline

This repository contains three Python scripts that set up a LanceDB-powered retrieval augmented generation (RAG) workflow backed by a llama.cpp model.

## Project Structure

- `initializer.py` – downloads required binaries, models, and verifies library availability.
- `ingest.py` – indexes Markdown documents into a LanceDB database with embeddings.
- `chatbot.py` – runs an interactive chatbot that answers questions using the indexed knowledge base.
- `requirements.txt` – lists Python dependencies needed by the scripts.

## Quick Start

1. **Install dependencies**
   ```bash
   python -m pip install -r requirements.txt
   ```

2. **Run the initializer** (downloads LanceDB model assets, llama.cpp, and GPT-OSS 20B model):
   ```bash
   python initializer.py
   ```

3. **Ingest Markdown files**:
   ```bash
   python ingest.py
   ```
   When prompted, provide the absolute or relative path to the directory containing `.md` files. The script stores embeddings in a `mydata` folder created alongside the provided directory.

4. **Start the chatbot**:
   ```bash
   python chatbot.py <path-to-your-markdown-directory>
   ```
   Ensure the path matches the one used during ingestion so the chatbot can open the existing LanceDB database.

## Notes

- The scripts print concise progress updates and terminate with helpful error messages when prerequisites are missing.
- Downloads performed by `initializer.py` may be large. Make sure adequate disk space and bandwidth are available.
- The ingestion script chunks documents to approximately 500 tokens with 50-token overlap and stores metadata such as source file paths and line ranges to aid in traceability.
