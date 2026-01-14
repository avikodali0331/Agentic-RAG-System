import concurrent.futures
import logging
import os
import shutil
from typing import List, Dict, Any

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from config import DEFAULT_CONFIG, AppConfig
from logging_config import setup_logging
from ingestion import load_single_file, split_documents
from vectorstore import (
    create_or_update_faiss,
    load_faiss,
    filter_new_files,
    update_manifest_with_files,
)
from tools import build_tools
from agent import build_agent

# Initialize logging immediately
setup_logging(DEFAULT_CONFIG.log_level)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Agentic RAG", page_icon="🕵️", layout="wide")
st.title("Agentic RAG with Ollama")
st.caption("Planner → Tool-Use → Critic (Retry Loop) → Final")

# --- State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# --- Helper Functions ---

def reset_persisted_store(persist_dir: str) -> None:
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)

def format_citations_from_chunks(chunks: List[Dict[str, Any]]) -> str:
    if not chunks: return ""
    unique_sources = {}
    for c in chunks:
        key = (c.get("source"), c.get("page"))
        if key not in unique_sources:
            unique_sources[key] = c.get("content", "")
    lines = ["**Referenced Sources:**"]
    for (src, page), content in list(unique_sources.items())[:6]:
        if page: lines.append(f"- *{src}* (p. {page})")
        else: lines.append(f"- *{src}*")
    return "\n".join(lines)

@st.cache_resource(show_spinner=False)
def get_vectorstore_cached(persist_dir: str, embedding_model: str, allow_dangerous: bool):
    return load_faiss(
        embedding_model=embedding_model, 
        persist_dir=persist_dir, 
        allow_dangerous=allow_dangerous
    )

def build_retriever(vs, k: int):
    return vs.as_retriever(search_kwargs={"k": k})

# --- Sidebar ---

def sidebar_config(default: AppConfig) -> AppConfig:
    st.sidebar.header("Settings")
    
    st.sidebar.subheader("Models")
    llm_options = ["llama3.1", "llama3.2", "mistral", "qwen2.5", "gemma2", "Other…"]
    embed_options = ["nomic-embed-text", "mxbai-embed-large", "bge-m3", "Other…"]

    llm_choice = st.sidebar.selectbox("LLM model", llm_options)
    llm_model = st.sidebar.text_input("Custom LLM", value=default.llm_model) if llm_choice == "Other…" else llm_choice

    embed_choice = st.sidebar.selectbox("Embedding model", embed_options)
    embedding_model = st.sidebar.text_input("Custom Embedding", value=default.embedding_model) if embed_choice == "Other…" else embed_choice

    st.sidebar.subheader("Generation")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, value=default.temperature, step=0.05)
    
    st.sidebar.subheader("Retrieval")
    k = st.sidebar.number_input("Top-k chunks", 1, 20, value=default.k)
    chunk_size = st.sidebar.number_input("Chunk size", 200, 4000, value=default.chunk_size)
    chunk_overlap = st.sidebar.number_input("Overlap", 0, 1000, value=default.chunk_overlap)
    persist_dir = st.sidebar.text_input("Persist directory", value=default.persist_dir)

    st.sidebar.subheader("Security")
    allow_dangerous = st.sidebar.checkbox(
        "Allow Local Index Deserialization", 
        value=default.allow_dangerous,
        help="Check this ONLY if you created the vector index yourself locally."
    )

    return AppConfig(
        llm_model=llm_model,
        embedding_model=embedding_model,
        temperature=float(temperature),
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        k=int(k),
        persist_dir=persist_dir,
        allow_dangerous=allow_dangerous,
        log_level=default.log_level,
    )

cfg = sidebar_config(DEFAULT_CONFIG)

with st.sidebar:
    st.header("1) Upload & Ingest")
    uploaded_files = st.file_uploader("Choose files", type=["pdf", "txt"], accept_multiple_files=True)
    
    colA, colB = st.columns(2)
    ingest_clicked = colA.button("Ingest", type="primary", use_container_width=True)
    load_clicked = colB.button("Load", use_container_width=True)
    
    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("Reset Index", type="secondary", use_container_width=True):
        reset_persisted_store(cfg.persist_dir)
        get_vectorstore_cached.clear()
        st.session_state.agent = None
        st.session_state.vectorstore_ready = False
        st.success("Index reset.")
        st.rerun()

    if load_clicked:
        vs = get_vectorstore_cached(cfg.persist_dir, cfg.embedding_model, cfg.allow_dangerous)
        if vs:
            st.session_state.retriever = build_retriever(vs, cfg.k)
            tools = build_tools(st.session_state.retriever)
            st.session_state.agent = build_agent(tools, cfg.llm_model, cfg.temperature)
            st.session_state.vectorstore_ready = True
            st.success("Loaded from disk.")
        else:
            if not cfg.allow_dangerous:
                st.error("Load failed. Enable 'Allow Local Index Deserialization' if this is your trusted index.")
            else:
                st.error("No index found or load failed.")

    if ingest_clicked and uploaded_files:
        files_data = [(f.name, f.read()) for f in uploaded_files]
        files_data, skipped = filter_new_files(files_data, cfg.persist_dir)
        
        if skipped: st.info(f"Skipping {len(skipped)} existing files.")
        
        if not files_data:
            st.warning("No new files to ingest.")
        else:
            with st.status("Ingesting...", expanded=True) as status:
                all_docs = []
                with concurrent.futures.ThreadPoolExecutor() as ex:
                    futures = {ex.submit(load_single_file, fd): fd[0] for fd in files_data}
                    for fut in concurrent.futures.as_completed(futures):
                        try:
                            docs = fut.result()
                            all_docs.extend(docs)
                        except Exception as e:
                            st.error(f"Failed to ingest {futures[fut]}: {e}")
                
                status.write(f"Loaded {len(all_docs)} documents.")
                chunks = split_documents(all_docs, cfg.chunk_size, cfg.chunk_overlap)
                status.write(f"Split into {len(chunks)} chunks.")
                
                try:
                    vs, stats = create_or_update_faiss(
                        chunks, 
                        cfg.embedding_model, 
                        cfg.persist_dir,
                        allow_dangerous=cfg.allow_dangerous
                    )
                    update_manifest_with_files(files_data, cfg.persist_dir)
                    get_vectorstore_cached.clear()
                    
                    st.session_state.retriever = build_retriever(vs, cfg.k)
                    tools = build_tools(st.session_state.retriever)
                    st.session_state.agent = build_agent(tools, cfg.llm_model, cfg.temperature)
                    st.session_state.vectorstore_ready = True
                    status.update(label="Ingestion Complete", state="complete", expanded=False)
                except ValueError as ve:
                    st.error(str(ve))

st.divider()

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

if query := st.chat_input("Ask a question..."):
    st.chat_message("user").write(query)
    history_obj = st.session_state.messages.copy()
    st.session_state.messages.append(HumanMessage(content=query))

    if not st.session_state.agent:
        st.error("Please load or ingest documents first.")
    else:
        with st.chat_message("assistant"):
            status_container = st.status("Agent thinking...", expanded=True)
            final_answer = ""
            evidence_used = []
            
            try:
                inputs = {"user_query": query, "chat_history": history_obj}
                for event in st.session_state.agent.stream(inputs):
                    if "planner" in event:
                        status_container.write(f"**Plan**: {event['planner'].get('subquestions', [])}")
                    if "executor" in event:
                        trace = event["executor"].get("tool_trace", [])
                        if trace: status_container.write(f"`{trace[-1]}`")
                    if "critic" in event:
                        status = event["critic"].get("critic_status", "Unknown")
                        notes = event["critic"].get("critic_notes", "")
                        if status == "RETRY": status_container.write(f"🔴 **Critic (Retry)**: {notes}")
                        else: status_container.write("🟢 **Critic**: Evidence Approved.")
                    if "final" in event:
                        final_answer = event["final"].get("final_answer", "")
                        evidence_used = event["final"].get("evidence_chunks", [])
                
                status_container.update(label="Complete", state="complete", expanded=False)
                if final_answer:
                    st.markdown(final_answer)
                    st.session_state.messages.append(AIMessage(content=final_answer))
                    if evidence_used:
                        with st.expander("Sources (Actual Evidence Used)"):
                            st.markdown(format_citations_from_chunks(evidence_used))
            except Exception as e:

                st.error(f"Error: {e}")
