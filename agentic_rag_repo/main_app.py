#  main_app.py  --> Core application logic and orchestration (Updated with RAGAS)

import uuid
import streamlit as st

from database import init_mongodb
from weaviate_client import init_weaviate, WeaviateHybridRetriever
from rag_utils import create_rag_agent
from auth_ui import show_auth_page
from chat_ui import render_sidebar, display_chat_messages, handle_chat_input
from styles import apply_custom_styles
from config import COLLECTION_EMBED_MAP

def initialize_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = str(uuid.uuid4())
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def show_main_app(db, weaviate_client):
    """Show main application interface"""
    st.markdown('<h1 class="main-header">⚖️ Legal RAG Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar with settings
    with st.sidebar:
        settings = render_sidebar(db)
    
    # Extract settings
    selected_model = settings['selected_model']
    selected_collection = settings['selected_collection']
    alpha = settings['alpha']
    top_k = settings['top_k']
    show_metadata = settings['show_metadata']
    temperature = settings['temperature']
    max_tokens = settings['max_tokens']
    enable_ragas = settings.get('enable_ragas', False)
    
    # Initialize RAG components
    try:
        # Create retriever with current settings
        retriever = WeaviateHybridRetriever(
            client=weaviate_client,
            collection_name=selected_collection,
            embedding_model_name=COLLECTION_EMBED_MAP[selected_collection]["model"],
            alpha=alpha,
            k=top_k
        )
        
        # Create RAG agent
        agent = create_rag_agent(
            retriever=retriever,
            selected_model=selected_model,
            temperature=temperature,
            max_tokens=max_tokens,
            alpha=alpha,
            top_k=top_k,
            show_metadata=show_metadata
        )
        
        if agent is None:
            st.error("Failed to initialize RAG system. Please check your configuration.")
            return
        
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return
    
    # Display chat messages with RAGAS support
    display_chat_messages(show_metadata, enable_ragas)
    
    # Handle chat input with RAGAS evaluation
    handle_chat_input(agent, db, show_metadata, enable_ragas)

def main():
    """Main application entry point"""
    # Page config
    st.set_page_config(
        page_title="Legal RAG Assistant", 
        page_icon="⚖️", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom styles
    apply_custom_styles()
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize connections
    db = init_mongodb()
    weaviate_client = init_weaviate()
    
    if db is None:
        st.error("⚠️ Database connection failed. Please check your MongoDB configuration.")
        return
    
    if not weaviate_client:
        st.error("⚠️ Weaviate connection failed. Please check your configuration.")
        return

    # Authentication check
    if not st.session_state.authenticated:
        show_auth_page(db)
        return
    
    # Show main app
    show_main_app(db, weaviate_client)

if __name__ == "__main__":
    main()