"""Ingest Markdown documents into a LanceDB database with vector embeddings."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
from sentence_transformers import SentenceTransformer
import lancedb
import re  # Make sure this is at the top with other imports

# Define paths
DATA_DIR = Path("data")
DB_PATH = Path("downloads/lancedb")
DB_TABLE_NAME = "documents"


@dataclass
class TokenSlice:
    tokenText: str
    lineNumber: int


@dataclass
class Chunk:
    text: str
    startLine: int
    endLine: int


def printStatus(message: str) -> None:
    print(f"[ingest] {message}")


def promptForDirectory() -> Path:
    """Prompt for a directory path until a valid one is provided."""
    while True:
        directoryInput = input("Enter directory containing Markdown files: ").strip()

        if not directoryInput:
            printStatus("Directory path cannot be empty. Please try again.")
            continue

        try:
            directoryPath = Path(directoryInput).expanduser().resolve()
            if directoryPath.exists() and directoryPath.is_dir():
                return directoryPath
            else:
                printStatus(f"Directory not found: {directoryPath}")
                printStatus("Please enter a valid directory path.")
        except Exception as e:
            printStatus(f"Error processing path: {e}")
            printStatus("Please enter a valid directory path.")


def readMarkdownFiles(directoryPath: Path) -> List[Path]:
    markdownFiles = sorted(directoryPath.rglob("*.md"))
    return markdownFiles


def buildTokenStream(text: str) -> List[TokenSlice]:
    import bisect

    lineOffsets = [0]
    for index, character in enumerate(text):
        if character == "\n":
            lineOffsets.append(index + 1)

    tokens: List[TokenSlice] = []
    pattern = re.compile(r"\S+|\n")
    for match in pattern.finditer(text):
        tokenText = match.group(0)
        position = match.start()
        lineIndex = bisect.bisect_right(lineOffsets, position) - 1
        lineNumber = lineIndex + 1
        tokens.append(TokenSlice(tokenText=tokenText, lineNumber=lineNumber))
    return tokens


def chunkTokens(tokens: Sequence[TokenSlice], chunkSize: int = 1000, overlap: int = 100) -> List[Chunk]:
    """Split tokens into overlapping chunks with sizes appropriate for BGE-M3 model."""
    chunks: List[Chunk] = []
    startIndex = 0
    totalTokens = len(tokens)

    while startIndex < totalTokens:
        endIndex = min(startIndex + chunkSize, totalTokens)
        chunkTokens = tokens[startIndex:endIndex]
        chunkText = "".join(token.tokenText if token.tokenText == "\n" else f"{token.tokenText} " for token in chunkTokens)
        chunkText = chunkText.strip()

        if not chunkText:
            startIndex += chunkSize - overlap
            continue

        startLine = chunkTokens[0].lineNumber
        endLine = chunkTokens[-1].lineNumber
        chunks.append(Chunk(text=chunkText, startLine=startLine, endLine=endLine))

        if endIndex == totalTokens:
            break
        startIndex = max(endIndex - overlap, startIndex + 1)

    return chunks


# Create a custom embedding function
def get_embedding_function():
    """Create a sentence transformer embedding function."""
    printStatus("Initializing embedding model...")
    model = SentenceTransformer("BAAI/bge-m3")

    class EmbeddingFunction:
        def embed(self, texts):
            return model.encode(texts)

    return EmbeddingFunction()


def sanitize_text(text: str) -> str:
    """Remove HTML tags and sanitize text for safe database storage."""
    # Replace HTML tags with their text content
    clean_text = re.sub(r'<[^>]*>', '', text)

    # Handle markdown tables by replacing them with plain text
    # Replace table header/row separators with newlines
    clean_text = re.sub(r'\|[\s-]*\|[\s-]*\|', '\n', clean_text)

    # Replace table cells with simplified format
    clean_text = re.sub(r'\|\s*([^|]*)\s*\|', r'\1. ', clean_text)

    # Replace any remaining pipe characters with spaces
    clean_text = clean_text.replace('|', ' ')

    # Replace multiple spaces with single space
    clean_text = re.sub(r'\s+', ' ', clean_text)

    return clean_text


def generateRecords(filePath: Path, embeddingFunction) -> List[dict]:
    try:
        content = filePath.read_text(encoding="utf-8")
        tokens = buildTokenStream(content)
        chunks = chunkTokens(tokens)

        records: List[dict] = []
        for chunk in chunks:
            # Sanitize the text before embedding it
            sanitized_text = sanitize_text(chunk.text)

            # Skip empty chunks after sanitization
            if not sanitized_text.strip():
                continue

            try:
                vector = embeddingFunction.embed([sanitized_text])[0]
                records.append({
                    "text": sanitized_text,  # Use sanitized text
                    "vector": vector,
                    "sourcePath": str(filePath),
                    "startLine": chunk.startLine,
                    "endLine": chunk.endLine,
                })
            except Exception as e:
                printStatus(f"Error embedding chunk from {filePath.name}, skipping: {e}")
                continue

        return records
    except Exception as e:
        printStatus(f"Error processing {filePath.name}, skipping file: {e}")
        return []


def upsertRecords(table, records: Sequence[dict]) -> None:
    if not records:
        return
    table.add(records)


def ingestDirectory(directoryPath: Path) -> None:
    markdownFiles = readMarkdownFiles(directoryPath)
    if not markdownFiles:
        raise RuntimeError("No Markdown files were found in the provided directory")

    # Save database in the script's directory rather than in the markdown files directory
    scriptDir = Path(__file__).parent.resolve()
    databasePath = scriptDir / "mydata"

    printStatus(f"Saving database to {databasePath}")

    # Connect to the database
    db = lancedb.connect(str(databasePath))
    tableName = "documents"

    table = None

    # Try to open the table if it exists, with robust error handling
    if tableName in db.table_names():
        printStatus(f"Dropping existing table '{tableName}' due to potential schema conflicts")
        db.drop_table(tableName)
        printStatus("Existing table dropped")

    # Always create a fresh table
    printStatus(f"Creating new table '{tableName}'")

    # Get embedding dimension by encoding a test string
    embeddingFunction = get_embedding_function()
    test_embedding = embeddingFunction.embed(["Test"])[0]

    # Create the first record to establish proper schema
    first_record = {
        "text": "Test text",
        "vector": test_embedding,
        "sourcePath": "test/path",
        "startLine": 1,
        "endLine": 1,
    }

    # Create table with the first record to establish schema
    table = db.create_table(tableName, data=[first_record])
    printStatus(f"Created new table '{tableName}' with proper schema")

    # Use our custom embedding function
    embeddingFunction = get_embedding_function()

    # Process all markdown files
    for filePath in markdownFiles:
        printStatus(f"Processing {filePath}")
        records = generateRecords(filePath, embeddingFunction)
        if not records:
            printStatus(f"Skipping empty content in {filePath}")
            continue

        # Add records to the table (which now definitely exists)
        upsertRecords(table, records)

    printStatus("Ingestion complete")


def runIngestWorkflow() -> None:
    """Run the ingest workflow with path from command line or prompt."""
    try:
        # Check if path is provided as command line argument
        if len(sys.argv) > 1:
            # Use the first argument as the directory path
            input_path = sys.argv[1]
            try:
                directoryPath = Path(input_path).expanduser().resolve()
                if not directoryPath.exists() or not directoryPath.is_dir():
                    printStatus(f"Error: Directory not found: {directoryPath}")
                    printStatus("Usage: python ingest.py [directory_path]")
                    sys.exit(1)
                printStatus(f"Using directory from command line: {directoryPath}")
            except Exception as e:
                printStatus(f"Error processing path from command line: {e}")
                printStatus("Usage: python ingest.py [directory_path]")
                sys.exit(1)
        else:
            # No command line argument, fall back to interactive prompt
            printStatus("No directory specified on command line.")
            printStatus("Usage: python ingest.py [directory_path]")
            printStatus("Falling back to interactive prompt...")
            directoryPath = promptForDirectory()

        ingestDirectory(directoryPath)
    except Exception as error:
        printStatus(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    runIngestWorkflow()