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


def printStatus(message: str) -> None:
    """Print a status message."""
    print(f"[chatbot] {message}")


def printAssistant(message: str) -> None:
    """Print a message from the assistant."""
    print(f"\n[Assistant]: {message}\n")


def get_embedding_function():
    """Create an embedding function using SentenceTransformer."""
    printStatus("Initializing embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    class EmbeddingFunction:
        def embed(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            return model.encode(texts)

    return EmbeddingFunction()


def load_llama_model() -> Optional[llama_cpp.Llama]:
    """Load the LLM model using llama.cpp."""
    script_dir = Path(__file__).parent.resolve()
    model_path = script_dir / DEFAULT_MODEL_PATH

    # If the default model doesn't exist, look for any GGUF model
    if not model_path.exists():
        printStatus(f"Default model not found at {model_path}")
        # Look for any .gguf file in the models directory
        models_dir = script_dir / "downloads" / "models"
        if models_dir.exists():
            gguf_files = list(models_dir.glob("*.gguf"))
            if gguf_files:
                model_path = gguf_files[0]
                printStatus(f"Found alternative model: {model_path}")
            else:
                printStatus("No GGUF models found. Please run initializer.py first")
                return None
        else:
            printStatus("Models directory not found. Please run initializer.py first")
            return None

    try:
        printStatus(f"Loading model from {model_path}...")
        model = llama_cpp.Llama(
            model_path=str(model_path),
            n_ctx=MAX_TOKENS,
            n_batch=8,
            verbose=False
        )
        printStatus("Model loaded successfully")
        return model
    except Exception as e:
        printStatus(f"Error loading model: {e}")
        return None


def connect_to_database():
    """Connect to the LanceDB database in the script's directory."""
    # Use script's directory for database location
    script_dir = Path(__file__).parent.resolve()
    database_path = script_dir / "mydata"

    if not database_path.exists():
        printStatus(f"Database not found at {database_path}")
        printStatus("Please run ingest.py first to create the database")
        sys.exit(1)

    printStatus(f"Connecting to database at {database_path}")
    return lancedb.connect(str(database_path))


def search_documents(query: str, db, embedding_function) -> List[Dict[str, Any]]:
    """Search for relevant document chunks based on query."""
    printStatus(f"Searching for: {query}")

    # Check if the documents table exists
    table_name = "documents"
    if table_name not in db.table_names():
        printStatus(f"Table '{table_name}' not found in database")
        return []

    # Open the table
    table = db.open_table(table_name)

    # Generate query embedding using the same function as during ingestion
    query_embedding = embedding_function.embed(query)[0]

    # Search for similar documents
    try:
        results = (
            table.search(query_embedding)
            .metric("cosine")
            .limit(TOP_K)
            .to_pandas()
            .to_dict("records")
        )
    except Exception as e:
        printStatus(f"Error during search: {e}")
        # Fallback to a simpler approach if pandas conversion fails
        results = []
        raw_results = table.search(query_embedding).metric("cosine").limit(TOP_K).to_list()
        for item in raw_results:
            results.append({k: v for k, v in item.items()})

    return results


def create_prompt_with_context(query: str, results: List[Dict[str, Any]]) -> str:
    """Create a prompt for the LLM with context from the retrieved documents."""
    context = ""

    # Format the retrieved chunks into context
    for i, result in enumerate(results, 1):
        source_path = result.get("sourcePath", "Unknown source")
        text = result.get("text", "No text available")

        # Extract just the filename from the path for brevity
        filename = Path(source_path).name

        context += f"[Document {i}: {filename}]\n{text}\n\n"

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

    return prompt


def generate_response(model: llama_cpp.Llama, prompt: str) -> str:
    """Generate a response from the LLM model based on the prompt."""
    printStatus("Generating response...")

    try:
        # Generate completion
        output = model.create_completion(
            prompt,
            max_tokens=1024,
            stop=["</assistant>", "<user>"],
            temperature=0.7,
            stream=False
        )

        response = output["choices"][0]["text"].strip()
        return response

    except Exception as e:
        printStatus(f"Error generating response: {e}")
        return "Sorry, I encountered an error while generating a response."


def chat_loop():
    """Run the main chat loop."""
    printStatus("Initializing chatbot...")

    # Connect to the database
    db = connect_to_database()

    # Load the embedding function
    embedding_function = get_embedding_function()

    # Load the LLM model
    model = load_llama_model()
    if model is None:
        printStatus("Continuing in retrieval-only mode (no AI generation)")
    else:
        printStatus("Ready with AI generation capabilities!")

    printStatus("Ready! Type 'exit' or 'quit' to end the conversation.")

    while True:
        user_input = input("\n[You]: ").strip()

        if user_input.lower() in ("exit", "quit", "bye"):
            printStatus("Goodbye!")
            break

        if not user_input:
            continue

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


def format_results(results: List[Dict[str, Any]]) -> str:
    """Format search results into a readable response."""
    if not results:
        return "I couldn't find any relevant information for your query."

    formatted_response = "Here's what I found (retrieval only mode):\n\n"

    for i, result in enumerate(results, 1):
        source_path = result.get("sourcePath", "Unknown source")
        text = result.get("text", "No text available")
        lines = f"Lines {result.get('startLine', '?')}-{result.get('endLine', '?')}"
