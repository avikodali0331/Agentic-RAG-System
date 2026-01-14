import json
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_core.documents import Document

def build_tools(retriever) -> List:
    
    def _format_docs_to_json(docs: List[Document]) -> str:
        results = []
        for d in docs:
            meta = d.metadata or {}
            page = meta.get("page_display") 
            if page is None:
                p_raw = meta.get("page")
                if isinstance(p_raw, int):
                    page = p_raw + 1
            
            results.append({
                "content": d.page_content,
                "source": meta.get("source", "unknown"),
                "page": page 
            })
        return json.dumps(results, indent=2)

    @tool
    def search_documents(query: str) -> str:
        """Search relevant factual excerpts. Returns JSON."""
        docs = retriever.invoke(query)
        return _format_docs_to_json(docs)

    @tool
    def extract_risks(query: str) -> str:
        """Find risks, downsides, or negative outcomes. Returns JSON."""
        augmented = f"risks downsides danger negative limitations of {query}"
        docs = retriever.invoke(augmented)
        return _format_docs_to_json(docs)

    @tool
    def extract_rewards(query: str) -> str:
        """Find benefits, upsides, or positive outcomes. Returns JSON."""
        augmented = f"benefits advantages rewards positive outcomes of {query}"
        docs = retriever.invoke(augmented)
        return _format_docs_to_json(docs)

    @tool
    def find_definitions(query: str) -> str:
        """Find definitions of terms. Returns JSON."""
        augmented = f"definition meaning explanation of term {query}"
        docs = retriever.invoke(augmented)
        return _format_docs_to_json(docs)

    return [search_documents, extract_risks, extract_rewards, find_definitions]