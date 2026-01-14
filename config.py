from dataclasses import dataclass
import os

@dataclass(frozen=True)
class AppConfig:
    # Models
    llm_model: str = "llama3.1"
    embedding_model: str = "nomic-embed-text"
    temperature: float = 0.0

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    k: int = 6

    # Persistence
    persist_dir: str = os.path.join("data", "vectorstore_faiss")
    
    # Safety
    allow_dangerous: bool = False  # Default to False for safety

    # App
    log_level: str = "INFO"

DEFAULT_CONFIG = AppConfig()