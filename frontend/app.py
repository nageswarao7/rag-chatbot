import streamlit as st
import requests
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(page_title="RAG Chat", page_icon="💬", layout="centered")

st.title("💬 RAG Chat Bot")
st.caption("Ask questions about your documents")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

# --- Sidebar: Document Ingestion ---
with st.sidebar:
    st.header("📄 Document Ingestion")
    
    # Show stats
    doc_count = 0
    backend_connected = False
    try:
        stats = requests.get(f"{API_URL}/stats", timeout=5).json()
        doc_count = stats.get("total_documents", 0)
        backend_connected = True
    except:
        pass
    
    if backend_connected:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.metric("Chunks Indexed", doc_count)
        with col2:
            if doc_count > 0:
                if st.button("🗑️", help="Delete all documents"):
                    try:
                        response = requests.delete(f"{API_URL}/clear", timeout=10)
                        if response.status_code == 200:
                            st.session_state.ingested_files = set()
                            st.success("Deleted!")
                            st.rerun()
                        else:
                            st.error(f"Delete failed: {response.text}")
                    except Exception as e:
                        st.error(f"Delete error: {e}")
    else:
        st.warning("Backend not connected")
    
    st.divider()
    
    # File uploader with unique key
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        accept_multiple_files=True, 
        type=["txt", "md", "pdf", "docx"],
        key="file_uploader"
    )
    
    # Only show ingest button if there are NEW files
    new_files = []
    if uploaded_files:
        for f in uploaded_files:
            file_key = f"{f.name}_{f.size}"
            if file_key not in st.session_state.ingested_files:
                new_files.append(f)
    
    if new_files:
        st.info(f"{len(new_files)} new file(s) ready to ingest")
        
        if st.button("Ingest Documents", type="primary"):
            progress = st.progress(0)
            for i, file in enumerate(new_files):
                try:
                    response = requests.post(
                        f"{API_URL}/ingest",
                        files={"file": (file.name, file.getvalue(), "application/octet-stream")},
                        timeout=60
                    )
                    if response.status_code == 200:
                        # Mark as ingested
                        file_key = f"{file.name}_{file.size}"
                        st.session_state.ingested_files.add(file_key)
                        result = response.json()
                        st.success(f"✅ {file.name} ({result.get('chunks_created', 0)} chunks)")
                    else:
                        st.error(f"❌ {file.name}: {response.text}")
                except Exception as e:
                    st.error(f"❌ {file.name}: {e}")
                progress.progress((i + 1) / len(new_files))
            
            st.rerun()  # Refresh to update stats
    elif uploaded_files:
        st.success("All files already ingested ✓")

# --- Chat Interface ---
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for src in msg["sources"]:
                    st.caption(f"• {src}")

# Chat input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Build history for API (exclude sources, last 10 messages)
                history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[-11:-1]  # Exclude current message
                ]
                
                response = requests.post(
                    f"{API_URL}/chat", 
                    json={"message": prompt, "history": history}, 
                    timeout=120
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No response")
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📚 Sources"):
                            for src in sources:
                                st.caption(f"• {src}")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

