"""Utility script to prepare dependencies required for the LanceDB + llama.cpp pipeline."""
from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen, urlretrieve

LANCE_DB_PACKAGE = "lancedb"
LLAMA_CPP_PACKAGE = "llama-cpp-python"

LLAMA_CPP_URLS = {
    "Linux": "https://github.com/ggerganov/llama.cpp/releases/download/b3539/llama-b3539-bin-linux-x86_64.zip",
    "Darwin": "https://github.com/ggerganov/llama.cpp/releases/download/b3539/llama-b3539-bin-macos-universal2.zip",
    "Windows": "https://github.com/ggerganov/llama.cpp/releases/download/b3539/llama-b3539-bin-win-x64.zip",
}

GPT_OSS_MODEL_URL = "https://huggingface.co/TheBloke/gpt-oss-20B-GGUF/resolve/main/gpt-oss-20b.Q4_K_M.gguf"

DOWNLOAD_DIRECTORY = Path("downloads")


class InitializationError(Exception):
    """Raised when the initializer fails to complete a mandatory step."""


def printStatus(message: str) -> None:
    """Print a concise status message."""

    print(f"[setup] {message}")


def runCommand(command: list[str]) -> None:
    """Run a subprocess command and raise an informative error on failure."""

    try:
        subprocess.run([sys.executable, "-m", *command], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise InitializationError(
            f"Command failed: {' '.join(error.cmd)}\nstdout: {error.stdout.decode()}\nstderr: {error.stderr.decode()}"
        ) from error


def ensurePackageInstalled(packageName: str) -> None:
    """Install a package with pip if it is unavailable."""

    try:
        __import__(packageName.replace("-", "_"))
        printStatus(f"Package '{packageName}' already available")
        return
    except ModuleNotFoundError:
        printStatus(f"Installing package '{packageName}'")

    runCommand(["pip", "install", packageName])
    __import__(packageName.replace("-", "_"))
    printStatus(f"Package '{packageName}' installed")


def ensureLanceDb() -> None:
    """Ensure LanceDB is installed and that the default embedding model is cached."""

    ensurePackageInstalled(LANCE_DB_PACKAGE)

    from lancedb.embeddings import get_default_embedding_function

    printStatus("Preparing LanceDB default embedding model")
    embeddingFunction = get_default_embedding_function()

    # Trigger a download (if needed) by embedding a trivial text sample.
    try:
        _ = embeddingFunction.embed(["Initialization probe"])
        printStatus("Default embedding model ready")
    except Exception as error:  # pylint: disable=broad-except
        raise InitializationError(f"Failed to initialize LanceDB embedding model: {error}") from error


def downloadFile(url: str, destination: Path, expectedSha256: Optional[str] = None) -> None:
    """Download a file if missing and verify its integrity if a checksum is provided."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        printStatus(f"File already downloaded: {destination.name}")
        if expectedSha256:
            verifyChecksum(destination, expectedSha256)
        return

    printStatus(f"Downloading {destination.name}")
    try:
        with urlopen(url) as response:  # nosec B310
            totalSize = int(response.headers.get("Content-Length", "0"))
        if totalSize and totalSize > 0:
            printStatus(f"Download size: {totalSize / (1024 * 1024):.2f} MB")
    except URLError:
        printStatus("Could not fetch file size; continuing with download")

    try:
        urlretrieve(url, destination)  # nosec B310
    except URLError as error:
        raise InitializationError(f"Failed to download from {url}: {error}") from error

    if expectedSha256:
        verifyChecksum(destination, expectedSha256)
    printStatus(f"Saved to {destination}")


def verifyChecksum(filePath: Path, expectedSha256: str) -> None:
    """Validate a file against a sha256 checksum."""

    hashObject = hashlib.sha256()
    with filePath.open("rb") as filePointer:
        for chunk in iter(lambda: filePointer.read(8192), b""):
            hashObject.update(chunk)
    checksum = hashObject.hexdigest()
    if checksum != expectedSha256:
        filePath.unlink(missing_ok=True)
        raise InitializationError(
            f"Checksum mismatch for {filePath.name}: expected {expectedSha256}, received {checksum}"
        )
    printStatus(f"Checksum verified for {filePath.name}")


def ensureLlamaCppBinary() -> None:
    """Download the llama.cpp prebuilt binary appropriate for the current platform."""

    systemName = platform.system()
    url = LLAMA_CPP_URLS.get(systemName)
    if not url:
        raise InitializationError(f"Unsupported platform for llama.cpp binary download: {systemName}")

    archiveName = Path(url).name
    archivePath = DOWNLOAD_DIRECTORY / archiveName

    downloadFile(url, archivePath)

    extractTarget = DOWNLOAD_DIRECTORY / "llama.cpp"
    if extractTarget.exists():
        printStatus("llama.cpp binary already extracted")
        return

    printStatus("Extracting llama.cpp binary archive")
    shutil.unpack_archive(str(archivePath), str(extractTarget))
    printStatus("llama.cpp binary ready")


def ensureLlamaCppPython() -> None:
    """Install llama-cpp-python module."""

    ensurePackageInstalled(LLAMA_CPP_PACKAGE)


def ensureGptOssModel() -> None:
    """Download the GPT-OSS 20B quantized model file."""

    modelFileName = Path(GPT_OSS_MODEL_URL).name
    modelPath = DOWNLOAD_DIRECTORY / "models" / modelFileName
    downloadFile(GPT_OSS_MODEL_URL, modelPath)


def runInitializer() -> None:
    """Run the entire initialization workflow."""

    try:
        ensureLanceDb()
        ensureLlamaCppBinary()
        ensureLlamaCppPython()
        ensureGptOssModel()
    except InitializationError as error:
        printStatus(str(error))
        sys.exit(1)

    printStatus("Initialization complete")


if __name__ == "__main__":
    runInitializer()
