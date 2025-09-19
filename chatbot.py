"""Chatbot that uses LanceDB for retrieval and llama.cpp for generation."""
from __future__ import annotations

import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import lancedb

# Check for required packages and install if missing
try:
    import pandas as pd
except ImportError:
    print("[chatbot] Installing pandas...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[chatbot] Installing sentence-transformers...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
    from sentence_transformers import SentenceTransformer

try:
    import llama_cpp
except ImportError:
    print("[chatbot] Installing llama-cpp-python...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
    import llama_cpp

# Number of relevant chunks to retrieve
TOP_K = 5
# Context size for the model
MAX_TOKENS = 2048
# Default model path (relative to the script directory)
DEFAULT_MODEL_PATH = "downloads/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
# Verbosity flag - set to True to see detailed information
VERBOSE = True


def printStatus(message: str) -> None:
    """Print a status message."""
    print(f"[chatbot] {message}")


def printVerbose(message: str) -> None:
    """Print verbose information if VERBOSE is enabled."""
    if VERBOSE:
        print(f"[chatbot:verbose] {message}")


def printAssistant(message: str) -> None:
    """Print a message from the assistant."""
    print(f"\n[Assistant]: {message}\n")


def get_embedding_function():
    """Create an embedding function using SentenceTransformer."""
    printStatus("Initializing embedding model...")
    printVerbose("Using SentenceTransformer with 'all-MiniLM-L6-v2' model")
    printVerbose("This model produces embeddings with 384 dimensions")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    class EmbeddingFunction:
        def embed(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            printVerbose(f"Generating embeddings for {len(texts)} text segments")
            return model.encode(texts)

    return EmbeddingFunction()


def load_llama_model() -> Optional[llama_cpp.Llama]:
    """Load the LLM model using llama.cpp."""
    script_dir = Path(__file__).parent.resolve()
    model_path = script_dir / DEFAULT_MODEL_PATH

    # If the default model doesn't exist, look for any GGUF model
    if not model_path.exists():
        printStatus(f"Default model not found at {model_path}")
        printVerbose("Searching for alternative GGUF models...")

        # Look for any .gguf file in the models directory
        models_dir = script_dir / "downloads" / "models"
        if models_dir.exists():
            gguf_files = list(models_dir.glob("*.gguf"))
            if gguf_files:
                model_path = gguf_files[0]
                printStatus(f"Found alternative model: {model_path}")
                printVerbose(f"Model file size: {model_path.stat().st_size / (1024*1024):.1f} MB")
            else:
                printStatus("No GGUF models found. Please run initializer.py first")
                printVerbose("GGUF models typically have a .gguf extension and are compatible with llama.cpp")
                return None
        else:
            printStatus("Models directory not found. Please run initializer.py first")
            printVerbose(f"Expected models directory at: {models_dir}")
            return None

    try:
        printStatus(f"Loading model from {model_path}...")
        printVerbose(f"Model file size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        printVerbose(f"Using context size of {MAX_TOKENS} tokens")

        load_start_time = time.time()
        model = llama_cpp.Llama(
            model_path=str(model_path),
            n_ctx=MAX_TOKENS,
            n_batch=8,
            verbose=VERBOSE,
        )
        load_duration = time.time() - load_start_time

        printStatus(f"Model loaded successfully in {load_duration:.2f} seconds")
        printVerbose(f"Model metadata: {model.model_path}")
        printVerbose(f"Vocabulary size: {model.n_vocab()} tokens")
        return model
    except Exception as e:
        printStatus(f"Error loading model: {e}")
        printVerbose("Try running initializer.py again to download a working model")
        return None


def connect_to_database():
    """Connect to the LanceDB database in the script's directory."""
    # Use script's directory for database location
    script_dir = Path(__file__).parent.resolve()
    database_path = script_dir / "mydata"

    if not database_path.exists():
        printStatus(f"Database not found at {database_path}")
        printStatus("Please run ingest.py first to create the database")
        printVerbose(f"Expected database directory at: {database_path}")
        sys.exit(1)

    printStatus(f"Connecting to database at {database_path}")
    db = lancedb.connect(str(database_path))

    # Show database info if verbose
    if VERBOSE:
        try:
            table_names = db.table_names()
            printVerbose(f"Available tables: {', '.join(table_names) or 'None'}")
            if "documents" in table_names:
                table = db.open_table("documents")
                stats = table.stats()
                printVerbose(f"Documents table contains {stats.get('num_rows', 'unknown')} rows")
        except Exception as e:
            printVerbose(f"Couldn't retrieve database stats: {e}")

    return db


def search_documents(query: str, db, embedding_function) -> List[Dict[str, Any]]:
    """Search for relevant document chunks based on query."""
    printStatus(f"Searching for: {query}")
    printVerbose(f"Query length: {len(query)} characters")

    # Check if the documents table exists
    table_name = "documents"
    if table_name not in db.table_names():
        printStatus(f"Table '{table_name}' not found in database")
        printVerbose("You need to ingest documents first using ingest.py")
        return []

    # Open the table
    table = db.open_table(table_name)
    printVerbose(f"Opened table '{table_name}'")

    # Generate query embedding using the same function as during ingestion
    printVerbose("Generating embedding for query")
    query_embedding = embedding_function.embed(query)[0]
    printVerbose(f"Embedding generated (dimension: {len(query_embedding)})")

    # Search for similar documents
    printVerbose(f"Searching for top {TOP_K} documents by cosine similarity")
    try:
        search_start_time = time.time()
        results = (
            table.search(query_embedding)
            .metric("cosine")
            .limit(TOP_K)
            .to_pandas()
            .to_dict("records")
        )
        search_duration = time.time() - search_start_time
        printVerbose(f"Search completed in {search_duration:.4f} seconds")
        printVerbose(f"Found {len(results)} matching documents")

        # Show similarity scores if available and verbose is on
        if VERBOSE and results:
            for i, result in enumerate(results):
                if "_distance" in result:
                    similarity = 1 - result["_distance"]  # Convert distance to similarity
                    printVerbose(f"Result {i+1} similarity score: {similarity:.4f}")

    except Exception as e:
        printStatus(f"Error during search: {e}")
        printVerbose("Falling back to simpler search method")
        # Fallback to a simpler approach if pandas conversion fails
        results = []
        raw_results = table.search(query_embedding).metric("cosine").limit(TOP_K).to_list()
        for item in raw_results:
            results.append({k: v for k, v in item.items()})
        printVerbose(f"Found {len(results)} matching documents using fallback method")

    return results


def create_prompt_with_context(query: str, results: List[Dict[str, Any]]) -> str:
    """Create a prompt for the LLM with context from the retrieved documents."""
    printVerbose(f"Creating prompt with {len(results)} document chunks as context")
    context = ""

    # Format the retrieved chunks into context
    for i, result in enumerate(results, 1):
        source_path = result.get("sourcePath", "Unknown source")
        text = result.get("text", "No text available")

        # Extract just the filename from the path for brevity
        filename = Path(source_path).name

        context += f"[Document {i}: {filename}]\n{text}\n\n"

    context_length = len(context)
    printVerbose(f"Context length: {context_length} characters")

    # Create the full prompt with system instructions
    prompt = f"""<system>
You are a helpful assistant. Answer the user's question based on the following document snippets.
If you can't answer based on the provided information, say so honestly.
</system>

<context>
{context}
</context>

<user>
{query}
</user>

<assistant>
"""

    printVerbose(f"Total prompt length: {len(prompt)} characters")
    return prompt


def generate_response(model: llama_cpp.Llama, prompt: str) -> str:
    """Generate a response from the LLM model based on the prompt."""
    printStatus("Generating response...")
    printVerbose("Sending prompt to language model")

    try:
        # Generate completion
        gen_start_time = time.time()
        output = model.create_completion(
            prompt,
            max_tokens=1024,
            stop=["</assistant>", "<user>"],
            temperature=0.7,
            stream=False
        )
        gen_duration = time.time() - gen_start_time

        response = output["choices"][0]["text"].strip()
        printVerbose(f"Response generated in {gen_duration:.2f} seconds")
        printVerbose(f"Response length: {len(response)} characters")
        return response

    except Exception as e:
        printStatus(f"Error generating response: {e}")
        printVerbose("Check if the model was loaded properly and has enough context")
        return "Sorry, I encountered an error while generating a response."


def chat_loop():
    """Run the main chat loop."""
    # Add import time for the verbose timing
    import time

    start_time = time.time()
    printStatus("Initializing chatbot...")

    # Connect to the database
    db = connect_to_database()

    # Load the embedding function
    embedding_function = get_embedding_function()

    # Load the LLM model
    model = load_llama_model()
    if model is None:
        printStatus("Continuing in retrieval-only mode (no AI generation)")
        printVerbose("Run initializer.py to download a language model for AI generation")
    else:
        printStatus("Ready with AI generation capabilities!")

    # Startup complete
    setup_duration = time.time() - start_time
    printStatus(f"Initialization completed in {setup_duration:.2f} seconds")
    printStatus("Ready! Type 'exit' or 'quit' to end the conversation.")

    while True:
        user_input = input("\n[You]: ").strip()

        if user_input.lower() in ("exit", "quit", "bye"):
            printStatus("Goodbye!")
            break

        if not user_input:
            continue

        query_start_time = time.time()

        # Search for relevant documents using the embedding function
        results = search_documents(user_input, db, embedding_function)

        if not results:
            printAssistant("I couldn't find any relevant information for your question.")
            continue

        # If model is available, use it to generate a response
        if model is not None:
            # Create prompt with context
            prompt = create_prompt_with_context(user_input, results)

            # Generate response
            response = generate_response(model, prompt)

            # Print the generated response
            printAssistant(response)
        else:
            # Fall back to just showing the chunks if model isn't available
            response = format_results(results)
            printAssistant(response)

        # Show query handling time
        query_duration = time.time() - query_start_time
        printVerbose(f"Query processed in {query_duration:.2f} seconds")
