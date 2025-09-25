#  chat_ui.py ---> Chat interface, sidebar and message handling (Updated with RAGAS)

import json
import uuid
import datetime
import streamlit as st

from database import save_chat_message, get_user_chats, get_chat_messages
from config import GROQ_MODELS, COLLECTION_EMBED_MAP
from ragas_integration import get_ragas_evaluator, display_ragas_metrics, show_ragas_settings

def render_sidebar(db):
    """Render the sidebar with user info, settings, and chat history"""
    st.markdown(f'<p class="sidebar-header">Welcome, {st.session_state.user["username"]}!</p>', unsafe_allow_html=True)
    
    if st.button("Logout", type="secondary"):
        st.session_state.clear()
        st.rerun()
    
    st.divider()
    
    st.markdown('<p class="sidebar-header">🤖 Model Configuration</p>', unsafe_allow_html=True)
    
    selected_model = st.selectbox(
        "Select LLM Model",
        list(GROQ_MODELS.keys()),
        format_func=lambda x: GROQ_MODELS[x],
        key="selected_model"
    )
    
    selected_collection = st.selectbox(
        "Select Knowledge Base",
        list(COLLECTION_EMBED_MAP.keys()),
        format_func=lambda x: f"{x}: {COLLECTION_EMBED_MAP[x]['description']}",
        key="selected_collection"
    )
    
    st.divider()
    
    # Advanced Settings
    st.markdown('<p class="sidebar-header">⚙️ Advanced Settings</p>', unsafe_allow_html=True)
    
    with st.expander("Retrieval Settings"):
        alpha = st.slider("Alpha (Hybrid Search)", 0.0, 1.0, 0.5, 0.1,
                        help="0=keyword search, 1=semantic search")
        top_k = st.slider("Top K Results", 1, 10, 3, 1)
        show_metadata = st.checkbox("Show Metadata", False)
    
    with st.expander("Generation Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        max_tokens = st.slider("Max Tokens", 500, 4000, 2000, 100)
    
    st.divider()
    
    # RAGAS Settings
    enable_ragas = show_ragas_settings()
    
    st.divider()
    
    # Chat History
    render_chat_history(db)
    st.divider()
    
    # Help & Tips
    render_help_section()
    
    return {
        'selected_model': selected_model,
        'selected_collection': selected_collection,
        'alpha': alpha,
        'top_k': top_k,
        'show_metadata': show_metadata,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'enable_ragas': enable_ragas
    }

def render_chat_history(db):
    """Render chat history section"""
    st.markdown('<p class="sidebar-header">💬 Chat History</p>', unsafe_allow_html=True)
    
    user_chats = get_user_chats(db, st.session_state.user["_id"])
    
    if st.button("➕ New Chat", type="primary"):
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    for chat in user_chats:
        chat_name = chat["latest_message"][:20] + "..." if len(chat["latest_message"]) > 20 else chat["latest_message"]
        if st.button(f"💬 {chat_name}", key=f"chat_{chat['_id']}"):
            st.session_state.current_chat_id = chat["_id"]
            st.session_state.messages = get_chat_messages(db, chat["_id"])
            st.rerun()

def render_help_section():
    """Render help and tips section"""
    st.markdown('<p class="sidebar-header">💡 Help & Tips</p>', unsafe_allow_html=True)
    
    with st.expander("Query Tips"):
        st.markdown("""
        **Effective Legal Queries:**
        - Be specific about legal concepts
        - Include relevant case details
        - Use legal terminology when possible
        - Ask about specific jurisdictions

        **Examples:**
        - "Contract breach remedies"
        - "Supreme Court precedent on privacy"
        - "Corporate liability standards"
        """)

    with st.expander("Search Settings Guide"):
        st.markdown("""
        **Alpha Setting:**
        - 0.0: Pure keyword search
        - 0.5: Balanced hybrid search
        - 1.0: Pure semantic search

        **Top K:** Number of document chunks to retrieve
        **Temperature:** Controls response creativity
        """)


def display_chat_messages(show_metadata=False, enable_ragas=False):
    """Display all chat messages with optional RAGAS metrics"""
    for message in st.session_state.messages:
        if message["message_type"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Show RAGAS metrics if enabled and available
                if enable_ragas and "ragas_scores" in message and message["ragas_scores"]:
                    display_ragas_metrics(message["ragas_scores"])
                
                # Show metadata if enabled and available
                if show_metadata and "metadata" in message and message["metadata"]:
                    display_metadata(message["metadata"])


def display_metadata(metadata):
    """Display document metadata in an expander"""
    with st.expander("📊 Document Metadata"):
        for i, meta in enumerate(metadata, 1):
            # Ensure metadata is a dict
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}
        
            st.json({
                f"Document {i}": {
                    "Case Title": meta.get("case_title", "N/A"),
                    "Court": meta.get("court", "N/A"),
                    "File": meta.get("file_name", "N/A"),
                    "Score": f"{meta.get('score', 0):.3f}",
                    "Chunk Index": meta.get("chunk_index", 0)
                }
            })

def handle_chat_input(agent, db, show_metadata=False, enable_ragas=False):
    """Handle chat input and generate responses with optional RAGAS evaluation"""
    if query := st.chat_input("Ask me about legal documents and cases..."):
        
        with st.chat_message("user"):
            st.write(query)
        
        st.session_state.messages.append({
            "message_type": "user",
            "content": query,
            "timestamp": datetime.datetime.now()
        })
        
        # Save user message to DB
        save_chat_message(
            db, 
            st.session_state.user["_id"],
            st.session_state.current_chat_id,
            "user",
            query
        )
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching legal documents..."):
                try:
                    # Reset metadata
                    if 'last_retrieval_metadata' in st.session_state:
                        del st.session_state.last_retrieval_metadata
                    
                    response = agent.invoke({"input": query})
                    response_text = response["output"]
                    
                    st.write(response_text)
                    
                    # Get metadata if available
                    metadata = st.session_state.get('last_retrieval_metadata', [])
                    
                    # RAGAS Evaluation (if enabled)
                    ragas_scores = {}
                    if enable_ragas and metadata:
                        with st.spinner("Evaluating response quality..."):
                            try:
                                evaluator = get_ragas_evaluator()
                                if evaluator.is_available():
                                    # Extract contexts from metadata
                                    contexts = []
                                    for meta in metadata:
                                        if isinstance(meta, dict) and 'content' in meta:
                                            contexts.append(meta['content'])
                                        elif hasattr(meta, 'get'):
                                            contexts.append(meta.get('text', str(meta)))
                                    
                                    if not contexts and 'last_retrieval_contexts' in st.session_state:
                                        contexts = st.session_state.get('last_retrieval_contexts', [])
                                    
                                    if contexts:
                                        ragas_scores = evaluator.evaluate_response(
                                            question=query,
                                            answer=response_text,
                                            contexts=contexts
                                        )
                                        
                                        if ragas_scores:
                                            display_ragas_metrics(ragas_scores)
                            
                            except Exception as e:
                                st.warning(f"RAGAS evaluation failed: {e}")
                                ragas_scores = {}
                    
                    # Show metadata if enabled
                    if show_metadata and metadata:
                        display_metadata(metadata)
                    
                    # Add assistant message
                    message_data = {
                        "message_type": "assistant",
                        "content": response_text,
                        "timestamp": datetime.datetime.now(),
                        "metadata": metadata
                    }
                    
                    # Add RAGAS scores if available
                    if ragas_scores:
                        message_data["ragas_scores"] = ragas_scores
                    
                    st.session_state.messages.append(message_data)
                    
                    # Save assistant message to DB
                    save_message_metadata = {"retrieval_metadata": metadata}
                    if ragas_scores:
                        save_message_metadata["ragas_scores"] = ragas_scores
                    
                    save_chat_message(
                        db,
                        st.session_state.user["_id"],
                        st.session_state.current_chat_id,
                        "assistant",
                        response_text,
                        save_message_metadata
                    )
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    
                    st.session_state.messages.append({
                        "message_type": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.datetime.now()
                    })