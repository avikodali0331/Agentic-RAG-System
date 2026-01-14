import hashlib
import json
import logging
import os
from typing import Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

FINGERPRINTS_FILE = "doc_fingerprints.json"
MANIFEST_FILE = "ingest_manifest.json"

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def build_embeddings(embedding_model: str) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=embedding_model)

def _fingerprints_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, FINGERPRINTS_FILE)

def _manifest_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, MANIFEST_FILE)

def file_hash(name: str, content: bytes) -> str:
    h = hashlib.sha256()
    h.update(content)
    return h.hexdigest()

def load_manifest(persist_dir: str) -> Dict[str, Dict]:
    path = _manifest_path(persist_dir)
    if not os.path.exists(path): return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception: return {}

def save_manifest(persist_dir: str, manifest: Dict[str, Dict]) -> None:
    path = _manifest_path(persist_dir)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception:
        logger.exception("Failed to save ingest manifest.")

def filter_new_files(files_data: List[Tuple[str, bytes]], persist_dir: str) -> Tuple[List[Tuple[str, bytes]], List[str]]:
    _ensure_dir(persist_dir)
    manifest = load_manifest(persist_dir)
    new_files = []
    skipped = []
    for name, content in files_data:
        fh = file_hash(name, content)
        if fh in manifest:
            skipped.append(name)
        else:
            new_files.append((name, content))
    return new_files, skipped

def update_manifest_with_files(files_data: List[Tuple[str, bytes]], persist_dir: str) -> None:
    _ensure_dir(persist_dir)
    manifest = load_manifest(persist_dir)
    for name, content in files_data:
        fh = file_hash(name, content)
        manifest[fh] = {"name": name, "size": len(content)}
    save_manifest(persist_dir, manifest)

# --- Chunk Dedupe ---
def _load_fingerprints(persist_dir: str) -> Set[str]:
    path = _fingerprints_path(persist_dir)
    if not os.path.exists(path): return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(str(x) for x in data) if isinstance(data, list) else set()
    except Exception: return set()

def _save_fingerprints(persist_dir: str, fps: Set[str]) -> None:
    path = _fingerprints_path(persist_dir)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sorted(list(fps)), f)
    except Exception: pass

def _chunk_fingerprint(doc: Document) -> str:
    text = (doc.page_content or "").strip()
    text_norm = " ".join(text.split())
    source = doc.metadata.get("source", "")
    page = str(doc.metadata.get("page_display") or "")
    # Robust fingerprinting
    raw = f"{source}|{page}|{text_norm}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# --- FAISS ---
def load_faiss(embedding_model: str, persist_dir: str, allow_dangerous: bool = False) -> Optional[FAISS]:
    embeddings = build_embeddings(embedding_model)
    if not os.path.exists(os.path.join(persist_dir, "index.faiss")):
        return None
    try:
        return FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=allow_dangerous,
        )
    except Exception as e:
        logger.error(f"FAISS load error: {e}")
        return None

def create_or_update_faiss(
    chunks: List[Document], 
    embedding_model: str, 
    persist_dir: str,
    allow_dangerous: bool = False
) -> Tuple[FAISS, Dict[str, int]]:
    
    _ensure_dir(persist_dir)
    embeddings = build_embeddings(embedding_model)
    seen_fps = _load_fingerprints(persist_dir)
    
    new_chunks = []
    new_fps = set()

    for d in chunks:
        fp = _chunk_fingerprint(d)
        if fp in seen_fps: continue
        new_chunks.append(d)
        new_fps.add(fp)

    # Attempt to load existing index respecting the safety flag
    existing_vs = load_faiss(embedding_model, persist_dir, allow_dangerous=allow_dangerous)

    if existing_vs:
        if new_chunks:
            existing_vs.add_documents(new_chunks)
        vs = existing_vs
    else:
        # Guard: If no existing index and no new chunks, we can't create anything.
        if not new_chunks:
            raise ValueError("No new content to index, and no existing index found.")
            
        vs = FAISS.from_documents(new_chunks, embeddings)

    vs.save_local(persist_dir)
    if new_fps:
        seen_fps.update(new_fps)
        _save_fingerprints(persist_dir, seen_fps)

    return vs, {
        "chunks_total": len(chunks),
        "chunks_new": len(new_chunks),
        "chunks_deduped": len(chunks) - len(new_chunks),
    }