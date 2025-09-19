"""Chatbot that uses the LanceDB vector database for answering questions."""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any
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

# Number of relevant chunks to retrieve
TOP_K = 5


def printStatus(message: str) -> None:
    """Print a status message."""
    print(f"[chatbot] {message}")


def printAssistant(message: str) -> None:
    """Print a message from the assistant."""
    print(f"\n[Assistant]: {message}\n")


def get_embedding_model():
    """Create a sentence transformer embedding model."""
    printStatus("Loading embedding model...")
    return SentenceTransformer("all-MiniLM-L6-v2")


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


def search_documents(query: str, db, model) -> List[Dict[str, Any]]:
    """Search for relevant document chunks based on query."""
    printStatus(f"Searching for: {query}")

    # Check if the documents table exists
    table_name = "documents"
    if table_name not in db.table_names():
        printStatus(f"Table '{table_name}' not found in database")
        return []

    # Open the table
    table = db.open_table(table_name)

    # Generate query embedding
    query_embedding = model.encode([query])[0]

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


def format_results(results: List[Dict[str, Any]]) -> str:
    """Format search results into a readable response."""
    if not results:
        return "I couldn't find any relevant information for your query."

    formatted_response = "Here's what I found:\n\n"

    for i, result in enumerate(results, 1):
        source_path = result.get("sourcePath", "Unknown source")
        text = result.get("text", "No text available")
        lines = f"Lines {result.get('startLine', '?')}-{result.get('endLine', '?')}"

        formatted_response += f"--- Result {i} ---\n"
        formatted_response += f"Source: {source_path} ({lines})\n"
        formatted_response += f"{text}\n\n"

    return formatted_response


def chat_loop():
    """Run the main chat loop."""
    printStatus("Initializing chatbot...")

    # Connect to the database
    db = connect_to_database()

    # Load the embedding model
    model = get_embedding_model()

    printStatus("Ready! Type 'exit' or 'quit' to end the conversation.")

    while True:
        user_input = input("\n[You]: ").strip()

        if user_input.lower() in ("exit", "quit", "bye"):
            printStatus("Goodbye!")
            break

        if not user_input:
            continue

        # Search for relevant documents
        results = search_documents(user_input, db, model)

        # Format and display the response
        response = format_results(results)
        printAssistant(response)


if __name__ == "__main__":
    try:
        chat_loop()
    except KeyboardInterrupt:
        printStatus("\nGoodbye!")
    except Exception as e:
        printStatus(f"Error: {e}")
        sys.exit(1)
