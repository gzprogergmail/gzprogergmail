"""Chatbot that uses LanceDB for retrieval and llama.cpp for generation."""
from __future__ import annotations

import sys
import subprocess
import json
import time
import traceback
import logging  # Added for file logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import lancedb
import os  # Add this import at the top of the file

# Set up logging to file
log_file = Path(__file__).parent / "chatbot_log.txt"
logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("chatbot")

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
MAX_TOKENS = 8192
# Default model path (relative to the script directory)
DEFAULT_MODEL_PATH = "downloads/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
# Verbosity flag - set to True to see detailed information
VERBOSE = True
# Set this to False to reduce model loading verbosity
MODEL_VERBOSE = False


def printStatus(message: str) -> None:
    """Print a status message."""
    print(f"[chatbot] {message}")
    logger.info(message)


def printVerbose(message: str) -> None:
    """Print verbose information if VERBOSE is enabled."""
    if VERBOSE:
        print(f"[chatbot:verbose] {message}")
    # Always log to file regardless of console verbosity
    logger.debug(message)


def printAssistant(message: str) -> None:
    """Print a message from the assistant."""
    print(f"\n[Assistant]: {message}\n")
    logger.info(f"RESPONSE: {message}")


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
    print("[chatbot] Starting model load process...")
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
        print(f"[chatbot] This may take a while depending on model size...")
        printVerbose(f"Model file size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        print(f"[chatbot] Starting model initialization...")

        # Temporarily reduce console output
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        if not MODEL_VERBOSE:
            # Redirect output to suppress verbose llama.cpp initialization
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

        try:
            print("[chatbot] Loading model into memory...")
            load_start_time = time.time()
            model = llama_cpp.Llama(
                model_path=str(model_path),
                n_ctx=MAX_TOKENS,
                n_batch=8,
                verbose=False,  # Set to False to reduce verbosity
            )
            load_duration = time.time() - load_start_time
        finally:
            # Restore output streams
            if not MODEL_VERBOSE:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        printStatus(f"Model loaded successfully in {load_duration:.2f} seconds")
        print("[chatbot] Model is ready for use!")
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
    logger.info(f"QUESTION: {query}")

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

    # Log the results
    if results:
        logger.info(f"Found {len(results)} relevant chunks")
        for i, result in enumerate(results, 1):
            source = Path(result.get("sourcePath", "Unknown")).name
            logger.info(f"Chunk {i}: {source} (similarity: {1-result.get('_distance', 0):.4f})")
            logger.info(f"Content: {result.get('text', '')[:100]}...")
    else:
        logger.info("No relevant chunks found")

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

    # Log the full prompt
    logger.info(f"PROMPT: {prompt}")

    return prompt


def generate_response(model: llama_cpp.Llama, prompt: str) -> str:
    """Generate a response from the LLM model based on the prompt."""
    printStatus("Generating response...")

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

        # Log the response
        logger.info(f"MODEL RESPONSE: {response}")

        return response

    except Exception as e:
        printStatus(f"Error generating response: {e}")
        printVerbose("Check if the model was loaded properly and has enough context")
        return "Sorry, I encountered an error while generating a response."


def format_results(results: List[Dict[str, Any]]) -> str:
    """Format search results into a readable response."""
    if not results:
        return "I couldn't find any relevant information for your query."

    formatted_response = "Here's what I found (retrieval only mode):\n\n"

    for i, result in enumerate(results, 1):
        source_path = result.get("sourcePath", "Unknown source")
        text = result.get("text", "No text available")
        lines = f"Lines {result.get('startLine', '?')}-{result.get('endLine', '?')}"

        formatted_response += f"--- Result {i} ---\n"
        formatted_response += f"Source: {source_path} ({lines})\n"
        formatted_response += f"{text}\n\n"

    return formatted_response


def test_model_response(model: llama_cpp.Llama) -> None:
    """Test the model with a simple prompt to diagnose response quality."""
    printStatus("Running model test with a simple prompt...")

    # Try a simple prompt first
    simple_prompt = "Hi there, how are you?"
    printStatus(f"Testing with prompt: '{simple_prompt}'")

    try:
        simple_response = model.create_completion(
            simple_prompt,
            max_tokens=50,
            temperature=0.7,
            stream=False
        )
        print("\n[Model Test] Simple prompt:")
        print(f"Prompt: '{simple_prompt}'")
        print(f"Response: '{simple_response['choices'][0]['text']}'")

        # Try a formatted prompt that matches our template
        formatted_prompt = """<system>
You are a helpful assistant.
</system>

<user>
Hi there, how are you?
</user>

<assistant>
"""
        printStatus("Testing with formatted prompt in expected template format...")
        formatted_response = model.create_completion(
            formatted_prompt,
            max_tokens=50,
            stop=["</assistant>", "<user>"],
            temperature=0.7,
            stream=False
        )
        print("\n[Model Test] Formatted prompt:")
        print(f"Response: '{formatted_response['choices'][0]['text']}'")

        # Compare and suggest fixes
        if len(formatted_response['choices'][0]['text'].strip()) > len(simple_response['choices'][0]['text'].strip()):
            printStatus("The model responds better to the formatted prompt with proper tags.")
        else:
            printStatus("The model doesn't seem to be specialized for the current prompt format.")

        printStatus("Model test complete. If responses are poor, consider:")
        printStatus("1. Using a larger or different model")
        printStatus("2. Adjusting the prompt format")
        printStatus("3. Reducing context length")
        printStatus("4. Setting a different temperature (lower for more focused responses)")

    except Exception as e:
        printStatus(f"Error during model test: {e}")


def chat_loop():
    """Main loop to process user queries and provide responses."""
    printStatus("Chatbot is ready! Type 'quit' to exit.")

    # Load the LLM model
    model = None
    while model is None:
        try:
            model = load_llama_model()
            if model is None:
                printStatus("Retrying model load in 5 seconds...")
                time.sleep(5)  # Wait before retrying
        except Exception as e:
            printStatus(f"Error loading model: {e}")
            printVerbose(traceback.format_exc())
            printStatus("Retrying model load in 5 seconds...")
            time.sleep(5)  # Wait before retrying

    # Run simple test to diagnose model quality
    test_model_response(model)

    # Connect to the database
    db = connect_to_database()

    # Initialize the embedding function
    embedding_function = get_embedding_function()

    # Main query processing loop
    while True:
        try:
            user_input = input("\n[You]: ").strip()

            if user_input.lower() == "quit":
                printStatus("Goodbye!")
                logger.info("Chat session ended by user")
                break

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
        except Exception as query_error:
            printStatus(f"Error processing query: {query_error}")
            printVerbose(traceback.format_exc())
            logger.error(f"Error processing query: {query_error}", exc_info=True)
            printStatus("Ready for next query...")

    # Cleanup actions if needed
    printStatus("Cleaning up resources...")
    # Close database connection
    try:
        db.close()
        printStatus("Database connection closed")
    except Exception as e:
        printStatus(f"Error closing database connection: {e}")

    # Model cleanup (if applicable)
    try:
        if model is not None:
            model.cleanup()
            printStatus("Model resources cleaned up")
    except Exception as e:
        printStatus(f"Error cleaning up model resources: {e}")

    printStatus("Goodbye!")


# Measure startup time
start_time = time.time()
printStatus(f"Startup complete in {time.time() - start_time:.2f} seconds")

# Main entry point - provide immediate feedback
if __name__ == "__main__":
    print("\n[chatbot] Starting chatbot...")
    print("[chatbot] Initializing components...")

    start_time = time.time()

    try:
        print("[chatbot] Setting up logging...")
        # Logging already set up at module level

        print("[chatbot] Starting chat loop...")
        chat_loop()

    except KeyboardInterrupt:
        printStatus("\nGoodbye!")
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

    print(f"[chatbot] Total runtime: {time.time() - start_time:.2f} seconds")
