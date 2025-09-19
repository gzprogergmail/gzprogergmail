"""Ingest Markdown documents into a LanceDB database with vector embeddings."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
from sentence_transformers import SentenceTransformer
import lancedb

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
    import re

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


def chunkTokens(tokens: Sequence[TokenSlice], chunkSize: int = 500, overlap: int = 50) -> List[Chunk]:
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
    model = SentenceTransformer("all-MiniLM-L6-v2")

    class EmbeddingFunction:
        def embed(self, texts):
            return model.encode(texts)

    return EmbeddingFunction()


def generateRecords(filePath: Path, embeddingFunction) -> List[dict]:
    content = filePath.read_text(encoding="utf-8")
    tokens = buildTokenStream(content)
    chunks = chunkTokens(tokens)

    records: List[dict] = []
    for chunk in chunks:
        vector = embeddingFunction.embed([chunk.text])[0]
        records.append(
            {
                "text": chunk.text,
                "vector": vector,
                "sourcePath": str(filePath),
                "startLine": chunk.startLine,
                "endLine": chunk.endLine,
            }
        )
    return records


def upsertRecords(table, records: Sequence[dict]) -> None:
    if not records:
        return
    table.add(records)


def ingestDirectory(directoryPath: Path) -> None:
    markdownFiles = readMarkdownFiles(directoryPath)
    if not markdownFiles:
        raise RuntimeError("No Markdown files were found in the provided directory")

    databasePath = directoryPath / "mydata"
    # Fix: Use lancedb.connect instead of connect
    db = lancedb.connect(str(databasePath))
    tableName = "documents"
    table = db.open_table(tableName) if tableName in db.table_names() else None
    if table is not None:
        try:
            table.create_index(metric="cosine", vector_column_name="vector")
        except Exception:  # pylint: disable=broad-except
            pass

    # Fix: Use our custom embedding function instead of get_default_embedding_function
    embeddingFunction = get_embedding_function()

    for filePath in markdownFiles:
        printStatus(f"Processing {filePath}")
        records = generateRecords(filePath, embeddingFunction)
        if not records:
            printStatus(f"Skipping empty content in {filePath}")
            continue
        if table is None:
            table = db.create_table(tableName, data=records)
            try:
                table.create_index(metric="cosine", vector_column_name="vector")
            except Exception:  # pylint: disable=broad-except
                pass
        else:
            upsertRecords(table, records)

    printStatus("Ingestion complete")


def runIngestWorkflow() -> None:
    try:
        directoryPath = promptForDirectory()
        ingestDirectory(directoryPath)
    except Exception as error:  # pylint: disable=broad-except
        printStatus(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    runIngestWorkflow()
