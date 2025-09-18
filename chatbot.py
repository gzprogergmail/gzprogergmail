"""Interactive chatbot that uses LanceDB for retrieval augmented generation with llama.cpp."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from lancedb import connect
from lancedb.embeddings import get_default_embedding_function
from llama_cpp import Llama

MODEL_PATH = Path("downloads/models/gpt-oss-20b.Q4_K_M.gguf")
DATABASE_SUBDIR = "mydata"
TABLE_NAME = "documents"


def printStatus(message: str) -> None:
    print(f"[chatbot] {message}")


def loadModel() -> Llama:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Please run initializer.py and ensure the download completed."
        )
    printStatus("Loading language model (this may take a moment)")
    model = Llama(model_path=str(MODEL_PATH), n_ctx=4096, n_threads=4)
    return model


def connectDatabase(contentDirectory: Path):
    databasePath = contentDirectory / DATABASE_SUBDIR
    if not databasePath.exists():
        raise FileNotFoundError(
            f"LanceDB database not found at {databasePath}. Run ingest.py for the target directory first."
        )
    db = connect(str(databasePath))
    return db.open_table(TABLE_NAME)


def runHybridSearch(table, queryText: str, embeddingFunction, topK: int = 5) -> List[dict]:
    vector = embeddingFunction.embed([queryText])[0]

    vectorResults: List[dict] = []
    keywordResults: List[dict] = []

    try:
        vectorResults = table.search(vector).metric("cosine").limit(topK).to_list()
    except Exception:  # pylint: disable=broad-except
        pass

    try:
        keywordResults = table.search(queryText).limit(topK).to_list()
    except Exception:  # pylint: disable=broad-except
        pass

    combined: dict[tuple[str, int, int], dict] = {}
    scoreBias = 1.0

    for rank, item in enumerate(vectorResults, start=1):
        score = 1 / rank
        key = (item.get("sourcePath"), item.get("startLine"), item.get("endLine"))
        if key in combined:
            combined[key]["score"] += score * scoreBias
        else:
            combined[key] = {
                "score": score * scoreBias,
                "item": item,
            }

    for rank, item in enumerate(keywordResults, start=1):
        key = (item["sourcePath"], item["startLine"], item["endLine"])
        score = 1 / rank
        if key in combined:
            combined[key]["score"] += score
        else:
            combined[key] = {"score": score, "item": item}

    ranked = sorted(combined.values(), key=lambda entry: entry["score"], reverse=True)
    return [entry["item"] for entry in ranked[:topK]]


def buildPrompt(query: str, contexts: List[dict]) -> str:
    contextBlocks = []
    for item in contexts:
        block = (
            f"Source: {item.get('sourcePath')} (lines {item.get('startLine')} - {item.get('endLine')})\n"
            f"Content:\n{item.get('text')}"
        )
        contextBlocks.append(block)

    contextText = "\n\n".join(contextBlocks)
    prompt = (
        "You are a concise assistant. Use the provided context to answer the question. "
        "If the answer is not contained within the context, say you do not know.\n\n"
        f"Context:\n{contextText}\n\nQuestion: {query}\nAnswer:"
    )
    return prompt


def startChatLoop(table, model: Llama) -> None:
    embeddingFunction = get_default_embedding_function()

    while True:
        try:
            userInput = input("Ask a question (or type 'exit'): ").strip()
        except EOFError:
            print()
            break

        if not userInput:
            continue
        if userInput.lower() in {"exit", "quit"}:
            break

        contexts = runHybridSearch(table, userInput, embeddingFunction)
        if not contexts:
            printStatus("No relevant context found. Consider ingesting more content.")
            continue

        prompt = buildPrompt(userInput, contexts)
        try:
            response = model(prompt, max_tokens=512, stop=["Question:"])
        except Exception as error:  # pylint: disable=broad-except
            printStatus(f"Model inference failed: {error}")
            continue

        text = response.get("choices", [{}])[0].get("text", "").strip()
        print(f"\n{text}\n")

    printStatus("Goodbye")


def runChatbot() -> None:
    if len(sys.argv) < 2:
        print("Usage: python chatbot.py <path-to-ingested-directory>")
        sys.exit(1)

    contentDirectory = Path(sys.argv[1]).expanduser().resolve()
    try:
        table = connectDatabase(contentDirectory)
        model = loadModel()
    except Exception as error:  # pylint: disable=broad-except
        printStatus(f"Error: {error}")
        sys.exit(1)

    startChatLoop(table, model)


if __name__ == "__main__":
    runChatbot()
