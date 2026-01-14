import logging
import os
import tempfile
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".pdf", ".txt"}

def load_single_file(file_data: Tuple[str, bytes]) -> List[Document]:
    name, content = file_data
    ext = os.path.splitext(name)[1].lower()

    if ext not in SUPPORTED_EXTS:
        logger.warning("Skipping unsupported: %s", name)
        return []

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            docs = PyPDFLoader(tmp_path).load()
        else:
            docs = TextLoader(tmp_path, encoding="utf-8").load()

        for d in docs:
            d.metadata = dict(d.metadata or {})
            d.metadata.update({"source": name, "file_ext": ext})
        return docs

    except Exception:
        logger.exception("Failed to load file: %s", name)
        return []
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

def split_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    for c in chunks:
        md = dict(c.metadata or {})
        page = md.get("page")
        if isinstance(page, int):
             md["page_display"] = page + 1
        else:
             md["page_display"] = None
             
        c.metadata = md

    return chunks