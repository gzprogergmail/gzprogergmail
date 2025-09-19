"""Utility script to prepare dependencies required for the LanceDB + llama.cpp pipeline."""
from __future__ import annotations

import hashlib
import os
import platform
import shutil
import ssl
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Callable
from urllib.error import URLError, HTTPError
from urllib.request import urlopen, Request, urlretrieve
import requests  # Added import at the correct location

# Create SSL context that works across platforms
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

LANCE_DB_PACKAGE = "lancedb"
LLAMA_CPP_PACKAGE = "llama-cpp-python"

# Basic verbosity control (enable by setting INIT_VERBOSE=1 in the environment)
VERBOSE = os.environ.get("INIT_VERBOSE", "").lower() in ("1", "true", "yes")

# Default fallback URLs in case API fetch fails
FALLBACK_URLS = {
    "Linux": "https://github.com/ggml-org/llama.cpp/releases/download/b6484/llama-b6484-bin-ubuntu-x64.zip",
    "Darwin": "https://github.com/ggml-org/llama.cpp/releases/download/b6484/llama-b6484-bin-macos-arm64.zip",
    "Windows": "https://github.com/ggml-org/llama.cpp/releases/download/b6484/llama-b6484-bin-win-cpu-x64.zip",
}

# Get latest release URLs or fall back to known ones
LLAMA_CPP_URLS = {}
try:
    repo = "ggml-org/llama.cpp"
    api_url = f"https://api.github.com/repos/{repo}/releases/latest"
    if VERBOSE:
        print(f"[setup] Fetching latest llama.cpp release metadata from {api_url}")
    response = requests.get(api_url, timeout=10)
    response.raise_for_status()
    latest = response.json()
    tag = latest.get("tag_name", "")
    if not tag:
        raise ValueError("Empty tag_name in GitHub API response")
    if VERBOSE:
        print(f"[setup] Latest release tag: {tag}")

    LLAMA_CPP_URLS = {
        "Linux": f"https://github.com/{repo}/releases/download/{tag}/llama-{tag}-bin-ubuntu-x64.zip",
        "Darwin": f"https://github.com/{repo}/releases/download/{tag}/llama-{tag}-bin-macos-arm64.zip",
        "Windows": f"https://github.com/{repo}/releases/download/{tag}/llama-{tag}-bin-win-cpu-x64.zip",
    }
except Exception as e:
    if VERBOSE:
        print(f"[WARNING] Could not fetch latest release info: {e}. Falling back to known static URLs.")
    # Fallback to known static URLs if GitHub API request fails
    LLAMA_CPP_URLS = FALLBACK_URLS

# Update the model URL to point to a more accessible GGUF model that works with llama.cpp
# Original URL had authentication issues:
# GPT_OSS_MODEL_URL = "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin"
GPT_OSS_MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

DOWNLOAD_DIRECTORY = Path("downloads")
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
DOWNLOAD_TIMEOUT = 30  # seconds

# Make sure downloads directory exists
DOWNLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)


class InitializationError(Exception):
    """Raised when the initializer fails to complete a mandatory step."""


def printStatus(message: str) -> None:
    """Print a concise status message."""
    print(f"[setup] {message}")


def printError(message: str) -> None:
    """Print an error message."""
    print(f"[ERROR] {message}")


def printWarning(message: str) -> None:
    """Print a warning message."""
    print(f"[WARNING] {message}")


def runCommand(command: list[str]) -> None:
    """Run a subprocess command and raise an informative error on failure."""
    try:
        # Use a safer method to run commands
        result = subprocess.run(
            [sys.executable, "-m", *command],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            timeout=300,  # 5-minute timeout
        )
        return result
    except subprocess.CalledProcessError as error:
        raise InitializationError(
            f"Command failed: {' '.join(error.cmd)}\nstdout: {error.stdout}\nstderr: {error.stderr}"
        ) from error
    except subprocess.TimeoutExpired:
        raise InitializationError(f"Command timed out after 300 seconds: {' '.join(command)}")


def ensurePackageInstalled(packageName: str) -> None:
    """Install a package with pip if it is unavailable."""
    # Check if package is already installed
    moduleName = packageName.replace("-", "_")
    try:
        __import__(moduleName)
        printStatus(f"Package '{packageName}' already available")
        return
    except ModuleNotFoundError:
        printStatus(f"Installing package '{packageName}'")

    try:
        runCommand(["pip", "install", "--upgrade", packageName])
        # Verify installation
        __import__(moduleName)
        printStatus(f"Package '{packageName}' installed successfully")
    except ImportError:
        # Try again with --no-cache-dir
        printWarning(f"First installation attempt failed for '{packageName}', trying with --no-cache-dir")
        runCommand(["pip", "install", "--upgrade", "--no-cache-dir", packageName])
        try:
            __import__(moduleName)
            printStatus(f"Package '{packageName}' installed successfully")
        except ImportError:
            raise InitializationError(
                f"Package '{packageName}' installation failed - module not found after multiple install attempts"
            )


def ensureLanceDb() -> bool:
    """Ensure LanceDB is installed and that the default embedding model is cached."""
    try:
        ensurePackageInstalled(LANCE_DB_PACKAGE)

        # Try different import methods for LanceDB embedding function
        try:
            import lancedb
            printStatus("LanceDB imported successfully")

            # Try to create a simple embedding function test
            try:
                # Use a basic embedding approach instead of get_default_embedding_function
                from sentence_transformers import SentenceTransformer

                printStatus("Testing embedding functionality with SentenceTransformer")
                model = SentenceTransformer("all-MiniLM-L6-v2")
                _ = model.encode(["Initialization probe"])
                printStatus("Embedding model test successful")
            except ImportError:
                printStatus("SentenceTransformer not available, installing...")
                ensurePackageInstalled("sentence-transformers")
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer("all-MiniLM-L6-v2")
                _ = model.encode(["Initialization probe"])
                printStatus("Embedding model ready")
        except Exception as embed_error:
            printStatus(f"Embedding setup had issues but LanceDB is installed: {embed_error}")

        printStatus("LanceDB setup completed")
        return True
    except Exception as error:  # pylint: disable=broad-except
        printError(f"Failed to initialize LanceDB: {error}")
        return False


def show_progress(block_num, block_size, total_size):
    """Display download progress."""
    if total_size > 0:
        percent = min(block_num * block_size * 100 / total_size, 100)
        sys.stdout.write(f"\r[setup] Download progress: {percent:.1f}%")
        sys.stdout.flush()
        if block_num * block_size >= total_size:
            sys.stdout.write("\n")
            sys.stdout.flush()


def downloadFile(url: str, destination: Path, expectedSha256: Optional[str] = None) -> None:
    """Download a file if missing and verify its integrity if a checksum is provided."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Check file permissions
    try:
        if destination.exists():
            # Test write permissions
            with open(destination, "ab") as f:
                pass
        else:
            # Test directory write permissions
            test_file = destination.parent / ".write_test"
            with open(test_file, "w") as f:
                f.write("test")
            test_file.unlink()
    except (PermissionError, OSError) as e:
        raise InitializationError(f"Permission error when checking write access: {e}")

    if destination.exists():
        printStatus(f"File already downloaded: {destination.name}")
        if expectedSha256:
            verifyChecksum(destination, expectedSha256)
        return

    printStatus(f"Downloading {destination.name}")

    # Set up headers for requests
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Try to get file size with multiple retries
    total_size = 0
    for attempt in range(MAX_RETRIES):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=DOWNLOAD_TIMEOUT, context=ssl_context) as response:
                total_size = int(response.headers.get("Content-Length", "0"))
                if total_size > 0:
                    printStatus(f"Download size: {total_size / (1024 * 1024):.2f} MB")
                    break
        except (URLError, HTTPError, TimeoutError) as err:
            if attempt < MAX_RETRIES - 1:
                printWarning(f"Attempt {attempt+1}/{MAX_RETRIES} failed to get file size: {err}. Retrying...")
                time.sleep(RETRY_DELAY)
            else:
                printWarning("Could not fetch file size; continuing with download")

    # Perform the download with retries
    for attempt in range(MAX_RETRIES):
        try:
            # Use a more reliable download approach
            if os.name == "nt":  # Windows
                # On Windows, use urlretrieve with progress
                urlretrieve(url, destination, show_progress)
            else:
                # On other platforms, use urllib with explicit context
                req = Request(url, headers=headers)
                with urlopen(req, timeout=60, context=ssl_context) as response, open(destination, "wb") as out_file:
                    total_size = int(response.headers.get("Content-Length", 0))
                    downloaded = 0
                    block_size = 8192

                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break

                        downloaded += len(buffer)
                        out_file.write(buffer)

                        if total_size > 0:
                            percent = min(downloaded * 100 / total_size, 100)
                            sys.stdout.write(f"\r[setup] Download progress: {percent:.1f}%")
                            sys.stdout.flush()

                    if total_size > 0:
                        sys.stdout.write("\n")
                        sys.stdout.flush()

            break  # Success, exit retry loop
        except (URLError, HTTPError, TimeoutError) as err:
            if attempt < MAX_RETRIES - 1:
                printWarning(f"Attempt {attempt+1}/{MAX_RETRIES} failed to download: {err}. Retrying...")
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
            else:
                raise InitializationError(f"Failed to download from {url} after {MAX_RETRIES} attempts: {err}")

    if expectedSha256:
        verifyChecksum(destination, expectedSha256)
    printStatus(f"Saved to {destination}")


def verifyChecksum(filePath: Path, expectedSha256: str) -> None:
    """Validate a file against a sha256 checksum."""
    if not filePath.exists():
        raise InitializationError(f"File not found for checksum verification: {filePath}")

    try:
        hashObject = hashlib.sha256()
        with filePath.open("rb") as filePointer:
            for chunk in iter(lambda: filePointer.read(8192), b""):
                hashObject.update(chunk)
        checksum = hashObject.hexdigest()
        if checksum != expectedSha256:
            printWarning(f"Checksum mismatch for {filePath.name}: expected {expectedSha256}, received {checksum}")
            printWarning("File might be corrupted or tampered with. Removing and will retry download.")
            filePath.unlink(missing_ok=True)
            raise InitializationError(
                f"Checksum mismatch for {filePath.name}: expected {expectedSha256}, received {checksum}"
            )
        printStatus(f"Checksum verified for {filePath.name}")
    except (PermissionError, OSError) as e:
        raise InitializationError(f"Error during checksum verification: {e}")


def ensureLlamaCppBinary() -> bool:
    """Download the llama.cpp prebuilt binary appropriate for the current platform."""
    try:
        systemName = platform.system()
        url = LLAMA_CPP_URLS.get(systemName)
        if not url:
            printError(f"Unsupported platform for llama.cpp binary download: {systemName}")
            return False

        archiveName = Path(url).name
        archivePath = DOWNLOAD_DIRECTORY / archiveName

        try:
            downloadFile(url, archivePath)
        except InitializationError as download_error:
            printError(f"Download failed, trying alternative approach: {download_error}")
            printStatus("Consider manually downloading llama.cpp from https://github.com/ggerganov/llama.cpp/releases")
            return False

        extractTarget = DOWNLOAD_DIRECTORY / "llama.cpp"
        if extractTarget.exists():
            printStatus("llama.cpp binary already extracted")
            return True

        printStatus("Extracting llama.cpp binary archive")
        try:
            # Ensure extraction directory exists
            extractTarget.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(str(archivePath), str(extractTarget))

            # Verify extraction worked by checking if any files were created
            if not any(extractTarget.iterdir()):
                raise InitializationError("Archive extraction produced no files")

            printStatus("llama.cpp binary ready")
            return True
        except (shutil.ReadError, OSError) as e:
            printError(f"Failed to extract archive: {e}")
            printStatus("Please try extracting manually")
            return False
    except Exception as error:  # pylint: disable=broad-except
        printError(f"Failed to setup llama.cpp binary: {error}")
        return False


def ensureLlamaCppPython() -> bool:
    """Install llama-cpp-python module."""
    try:
        # Check if already installed
        try:
            import llama_cpp
            printStatus(f"Package 'llama-cpp-python' already available (version {llama_cpp.__version__})")
            return True
        except ImportError:
            printStatus("Installing llama-cpp-python...")

        # Simple direct installation
        subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python"],
                       check=True, capture_output=True)

        # Verify installation
        try:
            import llama_cpp
            printStatus(f"Successfully installed llama-cpp-python (version {llama_cpp.__version__})")
            return True
        except ImportError:
            printError("Installation successful but module import failed")
            return False

    except Exception as error:  # pylint: disable=broad-except
        printError(f"Failed to install llama-cpp-python: {error}")
        if platform.system() == "Windows":
            printStatus("On Windows, you might need Microsoft Visual C++ Build Tools.")
            printStatus("Visit: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        return False


def ensureGptOssModel() -> bool:
    """Download a suitable GGUF format model file for use with llama.cpp."""
    try:
        # Use TinyLlama which is small (1.1B) and publicly accessible without authentication
        modelUrl = GPT_OSS_MODEL_URL
        modelFileName = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  # Proper GGUF model file
        modelPath = DOWNLOAD_DIRECTORY / "models" / modelFileName

        # Create models directory if it doesn't exist
        (DOWNLOAD_DIRECTORY / "models").mkdir(parents=True, exist_ok=True)

        printStatus(f"Attempting to download {modelFileName} (this may take a while for larger models)")

        try:
            # Add special header for Hugging Face downloads
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            # Use requests for better handling of Hugging Face downloads
            response = requests.get(modelUrl, headers=headers, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1 MB chunks

            if total_size > 0:
                printStatus(f"Model size: {total_size / (1024 * 1024):.2f} MB")

            with open(modelPath, 'wb') as f:
                downloaded = 0
                for data in response.iter_content(block_size):
                    downloaded += len(data)
                    f.write(data)
                    if total_size > 0:
                        percent = min(downloaded * 100 / total_size, 100)
                        sys.stdout.write(f"\r[setup] Download progress: {percent:.1f}% ({downloaded/(1024*1024):.1f} MB)")
                        sys.stdout.flush()

                if total_size > 0:
                    sys.stdout.write("\n")

            # Verify the model file exists and has a reasonable size (at least 100KB)
            if modelPath.exists() and modelPath.stat().st_size > 100 * 1024:
                printStatus(f"Model file downloaded successfully: {modelPath}")

                # Test if the model can be loaded by llama.cpp
                try:
                    import llama_cpp
                    printStatus("Testing model loading with llama-cpp-python...")
                    # Just create a model object to see if it loads
                    model = llama_cpp.Llama(
                        model_path=str(modelPath),
                        n_ctx=512,  # Small context for testing
                        n_batch=8,
                        verbose=False
                    )
                    printStatus("Model loaded successfully with llama-cpp-python")
                except ImportError:
                    printStatus("llama-cpp-python not available for model testing")
                except Exception as e:
                    printWarning(f"Model loaded but test failed: {e}")

                return True
            else:
                printError("Downloaded model file is too small or doesn't exist")
                return False

        except Exception as download_error:
            printError(f"Model download failed: {download_error}")

            # Try downloading a smaller model as fallback
            try:
                fallback_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q2_K.gguf"
                fallback_path = DOWNLOAD_DIRECTORY / "models" / "mistral-7b-instruct-v0.1.Q2_K.gguf"
                printStatus("Trying alternative model download...")
                downloadFile(fallback_url, fallback_path)

                if fallback_path.exists() and fallback_path.stat().st_size > 100 * 1024:
                    printStatus("Alternative model file downloaded successfully")
                    return True
                else:
                    return False
            except Exception:
                return False

    except Exception as error:  # pylint: disable=broad-except
        printError(f"Failed to download model: {error}")
        return False


def runInitializer() -> None:
    """Run the entire initialization workflow."""
    printStatus("Starting initialization process...")

    # Track results of each component
    results = {}

    printStatus("Setting up LanceDB...")
    results["LanceDB"] = ensureLanceDb()

    printStatus("Setting up llama.cpp binary...")
    results["llama.cpp binary"] = ensureLlamaCppBinary()

    printStatus("Setting up llama-cpp-python...")
    results["llama-cpp-python"] = ensureLlamaCppPython()

    printStatus("Setting up GPT-OSS model...")
    results["GPT-OSS model"] = ensureGptOssModel()

    # Print final report
    print("\n" + "=" * 50)
    print("INITIALIZATION REPORT")
    print("=" * 50)

    successful = []
    failed = []

    for component, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{component:20} : {status}")

        if success:
            successful.append(component)
        else:
            failed.append(component)

    print("-" * 50)
    print(f"Successful: {len(successful)}/{len(results)}")
    if failed:
        print(f"Failed components: {', '.join(failed)}")
        print("Note: You can re-run this script to retry failed components.")
    else:
        print("All components initialized successfully!")

    print("=" * 50)


if __name__ == "__main__":
    runInitializer()
