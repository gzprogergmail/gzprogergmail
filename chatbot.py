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
# Context size for the model - will be dynamically adjusted
MAX_TOKENS_RATING = 2048  # Reduced for chunk rating phase
MAX_TOKENS_GENERATION = 8192  # For final answer generation
# Default model path (relative to the script directory) - changed to GPT-OSS
DEFAULT_MODEL_PATH = "downloads/models/gpt-oss-20b.Q4_K.gguf"
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
    printVerbose("Using SentenceTransformer with 'BAAI/bge-m3' model")
    printVerbose("This model produces embeddings with 1024 dimensions")
    model = SentenceTransformer("BAAI/bge-m3")

    class EmbeddingFunction:
        def embed(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            printVerbose(f"Generating embeddings for {len(texts)} text segments")
            return model.encode(texts)

    return EmbeddingFunction()


def load_llama_model(max_tokens: int = MAX_TOKENS_RATING) -> Optional[llama_cpp.Llama]:
    """Load the LLM model using llama.cpp with specified context size."""
    print(f"[chatbot] Starting model load process with {max_tokens} context size...")
    script_dir = Path(__file__).parent.resolve()
    model_path = script_dir / DEFAULT_MODEL_PATH

    # Check if the GPT-OSS model exists
    if not model_path.exists():
        printStatus(f"GPT-OSS model not found at {model_path}")
        printStatus("Please run initializer.py first to download the GPT-OSS model")
        printVerbose(f"Expected model file at: {model_path}")
        return None

    try:
        printStatus(f"Loading GPT-OSS model with {max_tokens} context size from {model_path}...")
        print(f"[chatbot] This may take a while for the larger GPT-OSS model...")
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
            print("[chatbot] Loading GPT-OSS model into memory...")
            load_start_time = time.time()
            model = llama_cpp.Llama(
                model_path=str(model_path),
                n_ctx=max_tokens,
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

        printStatus(f"GPT-OSS model loaded successfully in {load_duration:.2f} seconds")
        print(f"[chatbot] GPT-OSS model is ready with {max_tokens} context size!")
        return model
    except Exception as e:
        printStatus(f"Error loading GPT-OSS model: {e}")
        printVerbose("Try running initializer.py again to download the GPT-OSS model")
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
        # Generate completion with parameters tuned for GPT-OSS
        gen_start_time = time.time()
        output = model.create_completion(
            prompt,
            max_tokens=2048,  # Increased for GPT-OSS
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
        printVerbose("Check if the GPT-OSS model was loaded properly and has enough context")
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
    printStatus("Running GPT-OSS model test with a simple prompt...")

    # Try a simple prompt first
    simple_prompt = "Hi there, how are you?"
    printStatus(f"Testing GPT-OSS with prompt: '{simple_prompt}'")

    try:
        simple_response = model.create_completion(
            simple_prompt,
            max_tokens=100,  # Increased for GPT-OSS
            temperature=0.7,
            stream=False
        )
        print("\n[GPT-OSS Model Test] Simple prompt:")
        print(f"Prompt: '{simple_prompt}'")
        print(f"Response: '{simple_response['choices'][0]['text']}'")

        printStatus("GPT-OSS model test complete.")

    except Exception as e:
        printStatus(f"Error during GPT-OSS model test: {e}")


def rate_chunk_relevance(model: llama_cpp.Llama, query: str, chunk: Dict[str, Any]) -> tuple[int, bool]:
    """Rate how relevant a chunk is to the query using GPT-OSS (0-100) and suggest if whole document needed."""
    chunk_text = chunk.get("text", "")
    source_path = chunk.get("sourcePath", "Unknown")
    filename = Path(source_path).name
    start_line = chunk.get("startLine", "?")
    end_line = chunk.get("endLine", "?")

    # Create a concise rating prompt that requests ONLY JSON response
    rating_prompt = f"""Rate this document chunk's relevance to the question. Return ONLY JSON, no explanations.

Question: {query}

Chunk:
{chunk_text}

Return only: {{"rating": ..., "take_whole_document": ...}}

rating: 0-100
take_whole_document: true if need more context

JSON:"""

    try:
        printVerbose(f"Rating chunk from {filename} (lines {start_line}-{end_line})...")
        printVerbose(f"Chunk length: {len(chunk_text)} characters")

        rating_start_time = time.time()
        output = model.create_completion(
            rating_prompt,
            max_tokens=50,  # Reduced since we only want JSON
            temperature=0.0,  # Zero temperature for consistent JSON output
            stream=False
        )
        rating_duration = time.time() - rating_start_time

        response = output["choices"][0]["text"].strip()
        printVerbose(f"Rating completed in {rating_duration:.2f}s for {filename}")
        printVerbose(f"Raw response: {response}")

        # Parse JSON response - be more aggressive in extracting JSON
        import json
        import re

        # Clean the response - remove any non-JSON text
        json_match = re.search(r'\{[^}]*"rating"[^}]*"take_whole_document"[^}]*\}', response, re.DOTALL)
        if not json_match:
            # Try alternative pattern
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)

        if json_match:
            json_str = json_match.group(0).strip()
            printVerbose(f"Extracted JSON string: {json_str}")
            try:
                result = json.loads(json_str)
                rating = int(result.get("rating", 50))
                take_whole = bool(result.get("take_whole_document", False))

                # Clamp rating to valid range
                rating = max(0, min(100, rating))

                printVerbose(f"Parsed JSON successfully: rating={rating}, take_whole={take_whole}")
                logger.info(f"CHUNK RATING: {filename} lines {start_line}-{end_line}: {rating}/100, take_whole={take_whole}")

                return rating, take_whole

            except json.JSONDecodeError as e:
                printVerbose(f"JSON parse error: {e}")
                printVerbose(f"Raw JSON string was: {json_str}")

        # Enhanced fallback parsing
        printVerbose("JSON parsing failed, attempting enhanced fallback extraction...")

        # Try to extract rating number
        rating_match = re.search(r'"rating"\s*:\s*(\d{1,3})', response)
        take_whole_match = re.search(r'"take_whole_document"\s*:\s*(true|false)', response, re.IGNORECASE)

        if rating_match:
            rating = int(rating_match.group(1))
            rating = max(0, min(100, rating))
            take_whole = take_whole_match.group(1).lower() == "true" if take_whole_match else False

            printVerbose(f"Extracted via fallback: rating={rating}, take_whole={take_whole}")
            logger.info(f"CHUNK RATING (fallback): {filename}: {rating}/100, take_whole={take_whole}")
            return rating, take_whole

        # Last resort: extract any number
        numbers = re.findall(r'\b\d{1,3}\b', response)
        if numbers:
            rating = int(numbers[0])
            rating = max(0, min(100, rating))
            printVerbose(f"Extracted rating from number fallback: {rating}")
            logger.info(f"CHUNK RATING (number fallback): {filename}: {rating}/100")
            return rating, False

        printVerbose(f"Could not parse any rating from response: {response}")
        logger.info(f"CHUNK RATING (error): {filename}: 50/100 (default)")
        return 50, False

    except Exception as e:
        printVerbose(f"Error rating chunk from {filename}: {e}")
        logger.error(f"CHUNK RATING ERROR: {filename}: {e}")
        return 50, False


def search_and_rate_documents(query: str, db, embedding_function, rating_model: llama_cpp.Llama) -> List[Dict[str, Any]]:
    """Search for relevant documents and rate them using GPT-OSS."""
    printStatus(f"Searching and rating chunks for: {query}")
    logger.info(f"QUESTION: {query}")

    # First, get more chunks than we need for rating
    results = search_documents(query, db, embedding_function)

    if not results:
        return []

    # Print first chunk before AI rating - show the ENTIRE chunk
    if results:
        first_chunk = results[0]
        source_path = first_chunk.get("sourcePath", "Unknown")
        filename = Path(source_path).name
        chunk_text = first_chunk.get("text", "")
        start_line = first_chunk.get("startLine", "?")
        end_line = first_chunk.get("endLine", "?")

        print(f"\n[First Chunk Preview - {filename}]")
        print(f"Lines {start_line}-{end_line} | Length: {len(chunk_text)} characters")
        print("-" * 70)
        print(chunk_text)  # Show the ENTIRE chunk without truncation
        print("-" * 70)
        print()

    printStatus("Using GPT-OSS to rate chunk relevance...")
    printVerbose(f"Starting rating process for {len(results)} chunks")

    rating_start_time = time.time()

    # Rate each chunk using GPT-OSS
    rated_chunks = []
    for i, chunk in enumerate(results, 1):
        source = Path(chunk.get("sourcePath", "Unknown")).name
        printVerbose(f"Rating chunk {i}/{len(results)}: {source}")

        rating, take_whole = rate_chunk_relevance(rating_model, query, chunk)
        chunk["relevance_score"] = rating
        chunk["take_whole_document"] = take_whole
        rated_chunks.append(chunk)

        printVerbose(f"Chunk {i} rated: {rating}/100, take_whole={take_whole}")

    rating_duration = time.time() - rating_start_time
    printVerbose(f"Rating process completed in {rating_duration:.2f} seconds")

    # Sort by relevance score (highest first) and take top 2
    rated_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
    top_chunks = rated_chunks[:2]

    # Log the rating results with enhanced information
    printStatus("Chunk relevance ratings:")
    logger.info("RATING RESULTS:")
    for i, chunk in enumerate(rated_chunks, 1):
        source = Path(chunk.get("sourcePath", "Unknown")).name
        score = chunk.get("relevance_score", 0)
        take_whole = chunk.get("take_whole_document", False)
        selected = "✓" if chunk in top_chunks else " "
        take_whole_str = "📄" if take_whole else "📋"

        printStatus(f"{selected} Chunk {i}: {source} - Score: {score}/100 {take_whole_str}")
        logger.info(f"  Chunk {i}: {source} - Score: {score}/100, take_whole: {take_whole}, selected: {selected.strip()}")

    # Log summary statistics
    scores = [c.get("relevance_score", 0) for c in rated_chunks]
    avg_score = sum(scores) / len(scores) if scores else 0
    take_whole_count = sum(1 for c in rated_chunks if c.get("take_whole_document", False))

    printVerbose(f"Rating summary: avg={avg_score:.1f}, {take_whole_count}/{len(rated_chunks)} suggest whole document")
    logger.info(f"RATING SUMMARY: avg_score={avg_score:.1f}, take_whole_suggestions={take_whole_count}/{len(rated_chunks)}")
    logger.info(f"Selected top 2 chunks with scores: {[c.get('relevance_score', 0) for c in top_chunks]}")

    return top_chunks


def create_final_prompt(query: str, top_chunks: List[Dict[str, Any]]) -> str:
    """Create a prompt for final answer generation using the top-rated chunks."""
    printVerbose(f"Creating final prompt with {len(top_chunks)} top-rated chunks")

    context = ""
    for i, chunk in enumerate(top_chunks, 1):
        source_path = chunk.get("sourcePath", "Unknown source")
        text = chunk.get("text", "No text available")
        score = chunk.get("relevance_score", 0)
        take_whole = chunk.get("take_whole_document", False)
        filename = Path(source_path).name

        context += f"[Top Chunk {i}: {filename} (Relevance: {score}/100, Whole doc needed: {take_whole})]\n{text}\n\n"

    # Create the final prompt
    prompt = f"""You are a helpful assistant. Answer the user's question based on the following top-rated document chunks.
Provide a comprehensive and accurate answer based on the information provided.
If the information is insufficient to fully answer the question, mention what aspects you can address and what might be missing.

{context}

User Question: {query}

Answer:"""

    printVerbose(f"Final prompt length: {len(prompt)} characters")
    logger.info(f"FINAL PROMPT: {prompt}")

    return prompt


def chat_loop():
    """Main loop to process user queries and provide responses."""
    printStatus("Chatbot is ready! Type 'quit' to exit.")

    # Load the LLM model for rating (4096 context)
    printStatus("Loading GPT-OSS model for chunk rating...")
    rating_model = None
    while rating_model is None:
        try:
            rating_model = load_llama_model(MAX_TOKENS_RATING)
            if rating_model is None:
                printStatus("Retrying rating model load in 5 seconds...")
                time.sleep(5)
        except Exception as e:
            printStatus(f"Error loading rating model: {e}")
            printVerbose(traceback.format_exc())
            printStatus("Retrying rating model load in 5 seconds...")
            time.sleep(5)

    # Run simple test to diagnose model quality
    test_model_response(rating_model)

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

            # Stage 1: Search and rate chunks using GPT-OSS with 4096 context
            printStatus("Stage 1: Rating chunk relevance...")
            top_chunks = search_and_rate_documents(user_input, db, embedding_function, rating_model)

            if not top_chunks:
                printAssistant("I couldn't find any relevant information for your question.")
                continue

            # Stage 2: Reload GPT-OSS with 8192 context for final answer
            printStatus("Stage 2: Loading GPT-OSS with larger context for answer generation...")

            # Clean up the rating model
            try:
                if rating_model is not None:
                    rating_model.cleanup()
            except Exception as e:
                printVerbose(f"Error cleaning up rating model: {e}")

            # Load generation model with larger context
            generation_model = load_llama_model(MAX_TOKENS_GENERATION)

            if generation_model is not None:
                # Create final prompt with top chunks
                final_prompt = create_final_prompt(user_input, top_chunks)

                # Generate final response
                printStatus("Generating final answer...")
                response = generate_response(generation_model, final_prompt)

                # Print the generated response
                printAssistant(response)

                # Clean up generation model and reload rating model for next query
                try:
                    generation_model.cleanup()
                except Exception as e:
                    printVerbose(f"Error cleaning up generation model: {e}")

                # Reload rating model for next iteration
                printStatus("Reloading rating model for next query...")
                rating_model = load_llama_model(MAX_TOKENS_RATING)
            else:
                printAssistant("Sorry, I couldn't load the model for answer generation.")

            # Show query handling time
            query_duration = time.time() - query_start_time
            printVerbose(f"Query processed in {query_duration:.2f} seconds")

        except Exception as query_error:
            printStatus(f"Error processing query: {query_error}")
            printVerbose(traceback.format_exc())
            logger.error(f"Error processing query: {query_error}", exc_info=True)
            printStatus("Ready for next query...")

    # Cleanup actions
    printStatus("Cleaning up resources...")

    # Close database connection
    try:
        db.close()
        printStatus("Database connection closed")
    except Exception as e:
        printStatus(f"Error closing database connection: {e}")

    # Model cleanup
    try:
        if rating_model is not None:
            rating_model.cleanup()
            printStatus("Rating model resources cleaned up")
    except Exception as e:
        printStatus(f"Error cleaning up rating model: {e}")

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


