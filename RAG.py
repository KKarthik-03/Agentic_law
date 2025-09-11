import os
import time
import json
import datetime
import pandas as pd
import streamlit as st

from typing import List, Tuple, Optional
from dotenv import load_dotenv
import pymongo
import bcrypt
import weaviate
from sentence_transformers import SentenceTransformer

from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema.retriever import BaseRetriever
from langchain.prompts import PromptTemplate
from pydantic import Field, PrivateAttr
from bson import ObjectId

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f4e79;
        padding-bottom: 1rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 2rem;
    }
    
    .source-box {
        background: #f8f9fa;
        border-left: 4px solid #1f4e79;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Configuration & Setup
# -------------------------
load_dotenv()

# Environment variables
WEA_URL = os.environ.get("WEAVIATE_URL")
WEA_KEY = os.environ.get("WEAVIATE_API_KEY")
MONGO_URI = os.environ.get("MONGODB_URI")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Configuration constants
COLLECTION_EMBED_MAP = {
    "InLegalBERT_Chunks": {
        "model": "law-ai/InLegalBERT",
        "description": "Specialized legal domain model"
    },
    "Legal_Bert_Chunks": {
        "model": "nlpaueb/legal-bert-base-uncased",
        "description": "Legal BERT for case analysis"
    },
    "All_mpnet_Chunks": {
        "model": "all-mpnet-base-v2",
        "description": "General purpose sentence transformer"
    }
}

GROQ_MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B (Recommended)",
    "llama-3.1-8b-instant": "Llama 3.1 8B (Fast)",
    "openai/gpt-oss-20b": "GPT OSS 20B"
}

RELEVANCE_THRESHOLD = 0.70
MAX_CHAT_HISTORY = 50

# -------------------------
# Database Connections
# -------------------------
@st.cache_resource
def init_weaviate_client():
    """Initialize Weaviate client with caching."""
    if not WEA_URL or not WEA_KEY:
        return None
    try:
        if hasattr(weaviate, "connect_to_weaviate_cloud"):
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEA_URL,
                auth_credentials=weaviate.auth.AuthApiKey(WEA_KEY)
            )
        else:
            client = weaviate.Client(
                url=WEA_URL,
                additional_headers={"X-API-KEY": WEA_KEY}
            )
        return client
    except Exception as e:
        st.error(f"❌ Weaviate connection failed: {e}")
        return None

@st.cache_resource
def init_mongo_client():
    """Initialize MongoDB client with caching."""
    if not MONGO_URI:
        return None, None, None, None
    try:
        mongo_client = pymongo.MongoClient(MONGO_URI)
        db = mongo_client['legal_rag']
        users_col = db['users']
        chats_col = db['chat_history']
        return mongo_client, db, users_col, chats_col
    except Exception as e:
        st.error(f"❌ MongoDB connection failed: {e}")
        return None, None, None, None

# Initialize connections
weaviate_client = init_weaviate_client()
mongo_client, db, users_col, chats_col = init_mongo_client()

# -------------------------
# Custom Weaviate Hybrid Retriever
# -------------------------
class WeaviateHybridRetriever(BaseRetriever):
    client: weaviate.WeaviateClient = Field(...)
    collection_name: str = Field(...)
    alpha: float = Field(0.5)
    k: int = Field(5)
    embedding_model_name: str = Field(...)

    _embeddings: HuggingFaceEmbeddings = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        try:
            if self.client is None:
                return []
            
            collection = self.client.collections.get(self.collection_name)
            query_vector = self._embeddings.embed_query(query)
            
            results = collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=self.alpha,
                limit=self.k,
                return_properties=["text", "case_title", "court", "file_name", "chunk_index"],
                return_metadata=weaviate.classes.query.MetadataQuery(score=True)
            )
            
            documents = []
            for obj in results.objects:
                props = obj.properties or {}
                score = obj.metadata.score if getattr(obj, "metadata", None) and getattr(obj.metadata, "score", None) is not None else 0.0
                documents.append(Document(
                    page_content=props.get("text", ""),
                    metadata={
                        "case_title": props.get("case_title", ""),
                        "court": props.get("court", ""),
                        # "jurisdiction": props.get("jurisdiction", ""),
                        # "citation": props.get("citation", ""),
                        "file_name": props.get("file_name", ""),
                        "chunk_index": props.get("chunk_index", 0),
                        "score": score
                    }
                ))
            return documents
        except Exception as e:
            st.error(f"❌ Retrieval error: {e}")
            return []

    async def _aget_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)

# -------------------------
# Authentication Functions
# -------------------------
def hash_password(password: str) -> bytes:
    """Hash password with bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password: str, hashed: bytes) -> bool:
    """Verify password against hash."""
    try:
        return bcrypt.checkpw(password.encode(), hashed)
    except Exception:
        return False

def validate_credentials(username: str, password: str) -> Tuple[bool, str]:
    """Validate username and password format."""
    if len(username) < 3:
        return False, "Username must be at least 3 characters long"
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    if not username.replace('_', '').isalnum():
        return False, "Username can only contain letters, numbers, and underscores"
    return True, "Valid credentials"

def signup_user(username: str, password: str) -> Tuple[bool, str]:
    """Register new user."""
    if users_col is None:
        return False, "❌ Database not configured"
    
    # Validate credentials
    valid, msg = validate_credentials(username, password)
    if not valid:
        return False, f"❌ {msg}"
    
    try:
        if users_col.find_one({"username": username}):
            return False, "❌ Username already exists"
        
        hashed = hash_password(password)
        users_col.insert_one({
            "username": username,
            "password": hashed,
            "created_at": datetime.datetime.now(),
            "last_login": None
        })
        return True, "✅ Account created successfully!"
    except Exception as e:
        return False, f"❌ Registration failed: {e}"

def login_user(username: str, password: str) -> Tuple[bool, str]:
    """Authenticate user login."""
    if users_col is None:
        return False, "❌ Database not configured"
    
    try:
        user = users_col.find_one({"username": username})
        if user is None:
            return False, "❌ User not found"
        
        if not check_password(password, user.get('password', b"")):
            return False, "❌ Incorrect password"
        
        # Update last login
        users_col.update_one(
            {"username": username},
            {"$set": {"last_login": datetime.datetime.now()}}
        )
        
        return True, "✅ Login successful!"
    except Exception as e:
        return False, f"❌ Login failed: {e}"


# Session State Management

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'user': None,
        'chat_history': [],
        'summary_memory': "",
        'max_turns_direct': 1,
        'retriever': None,
        'llm': None,
        'collection_name': list(COLLECTION_EMBED_MAP.keys())[0],
        'llm_model': list(GROQ_MODELS.keys())[0],
        'show_metadata': True,
        'current_chat_id': None,
        'k_value': 3,
        'alpha_value': 0.5,
        'chat_sessions': [],
        'search_query': ""
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# RAG Helper Functions

def build_retriever_for_collection(collection_name: str, k: int = 3, alpha: float = 0.5):
    """Build retriever for specified collection."""
    if weaviate_client is None:
        st.error("❌ Cannot build retriever: Weaviate client not connected")
        return None
    
    model_info = COLLECTION_EMBED_MAP.get(collection_name)
    if not model_info:
        st.error(f"❌ No embedding model mapped for collection: {collection_name}")
        return None
    
    try:
        return WeaviateHybridRetriever(
            client=weaviate_client,
            collection_name=collection_name,
            alpha=alpha,
            k=k,
            embedding_model_name=model_info["model"]
        )
    except Exception as e:
        st.error(f"❌ Failed to build retriever: {e}")
        return None

def build_llm(model_name: str):
    """Build LLM with specified model."""
    if not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY not configured")
        return None
    
    try:
        return ChatGroq(
            model=model_name,
            groq_api_key=GROQ_API_KEY,
            temperature=0
        )
    except Exception as e:
        st.error(f"❌ Failed to build LLM: {e}")
        return None

def create_qa_chain(retriever, llm):
    """Create conversational retrieval chain with custom prompt."""
    system_template = """
    You are a Legal Document Assistant. Answer ONLY using the provided retrieved documents.
    
    RETRIEVED DOCUMENTS:
    {context}
    
    Chat History:
    {chat_history}
    
    USER QUESTION:
    {question}
    
    INSTRUCTIONS:
    1. Use ONLY information from retrieved documents
    2. Start with a concise answer (1-3 sentences)
    3. If no relevant information is found, respond: "I cannot find relevant information in the available legal documents."
    4. Be factual and neutral
    5. Add disclaimer: "This is document-based information only, not legal advice."
    
    Answer:
    """
    
    qa_prompt = PromptTemplate.from_template(system_template)
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

def summarize_old_history():
    """Summarize old chat history to manage context length."""
    max_direct = st.session_state.get('max_turns_direct', 3)
    chat_history = st.session_state.get('chat_history', [])
    
    if len(chat_history) <= max_direct:
        return
    
    old_history = chat_history[:-max_direct]
    recent_history = chat_history[-max_direct:]
    
    if not st.session_state.get('llm'):
        return
    
    old_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in old_history])
    
    try:
        summarize_chain = load_summarize_chain(st.session_state['llm'], chain_type="stuff")
        docs = [Document(page_content=old_text)]
        result = summarize_chain.invoke(docs)
        
        if isinstance(result, dict):
            summary = result.get('output_text', str(result))
        else:
            summary = str(result)
        
        st.session_state['summary_memory'] = summary
        st.session_state['chat_history'] = recent_history
        
    except Exception as e:
        st.warning(f"⚠️ Summarization failed: {e}")

# Database Operations
def save_chat_to_db(username: str):
    """Save current chat session to database."""
    if chats_col is None:
        return False
    
    try:
        chat_data = {
            'username': username,
            'timestamp': datetime.datetime.now(),
            'summary_memory': st.session_state.get('summary_memory', ""),
            'recent_turns': st.session_state.get('chat_history', []),
            'collection': st.session_state.get('collection_name', ""),
            'llm_model': st.session_state.get('llm_model', ""),
            'settings': {
                'k': st.session_state.get('k_value', 3),
                'alpha': st.session_state.get('alpha_value', 0.5)
            }
        }
        
        if st.session_state.get('current_chat_id'):
            chats_col.update_one(
                {'_id': ObjectId(st.session_state['current_chat_id'])},
                {'$set': chat_data}
            )
        else:
            result = chats_col.insert_one(chat_data)
            st.session_state['current_chat_id'] = str(result.inserted_id)
        
        return True
    except Exception as e:
        st.error(f"❌ Failed to save chat: {e}")
        return False

def load_user_chats(username: str, limit: int = 20) -> List[dict]:
    """Load user's chat history."""
    if chats_col is None:
        return []
    
    try:
        chats = list(
            chats_col.find({'username': username})
            .sort('timestamp', -1)
            .limit(limit)
        )
        return chats
    except Exception as e:
        st.error(f"❌ Failed to load chats: {e}")
        return []

def load_specific_chat(chat_id: str):
    """Load a specific chat session."""
    if chats_col is None:
        return False
    
    try:
        chat = chats_col.find_one({'_id': ObjectId(chat_id)})
        if chat:
            st.session_state['chat_history'] = chat.get('recent_turns', [])
            st.session_state['summary_memory'] = chat.get('summary_memory', "")
            st.session_state['current_chat_id'] = str(chat['_id'])
            
            # Load settings if available
            settings = chat.get('settings', {})
            st.session_state['k_value'] = settings.get('k', 5)
            st.session_state['alpha_value'] = settings.get('alpha', 0.5)
            
            return True
        return False
    except Exception as e:
        st.error(f"❌ Failed to load chat: {e}")
        return False

def delete_chat(chat_id: str):
    """Delete a specific chat session."""
    if chats_col is None:
        return False
    
    try:
        result = chats_col.delete_one({'_id': ObjectId(chat_id)})
        return result.deleted_count > 0
    except Exception as e:
        st.error(f"❌ Failed to delete chat: {e}")
        return False

# Query Processing

def process_query(query: str) -> Tuple[str, List[Document]]:
    """Process user query and return answer with sources."""
    # Check for blocklisted terms
    blocklist = ["weather", "recipe", "sports", "jokes", "cooking", "music"]
    if any(term in query.lower() for term in blocklist):
        return "I cannot find relevant information in the available legal documents. I am designed to answer legal queries only.", []
    
    # Ensure components are initialized
    retriever = st.session_state.get('retriever')
    llm = st.session_state.get('llm')
    
    if not retriever or not llm:
        return "❌ System not properly initialized. Please check settings.", []
    
    try:
        # Get relevant documents
        docs = retriever.invoke(query)
        if not docs:
            return "I cannot find relevant information in the available legal documents.", []
        
        # Check relevance threshold
        best_score = max([d.metadata.get("score", 0) for d in docs])
        if best_score < RELEVANCE_THRESHOLD:
            return "I cannot find sufficiently relevant information in the available legal documents.", []
        
        # Summarize old history if needed
        summarize_old_history()
        
        # Prepare chat history
        history_input = []
        if st.session_state.get('summary_memory'):
            history_input.append(("Summary", st.session_state['summary_memory']))
        history_input.extend(st.session_state.get('chat_history', []))
        
        # Create QA chain and get response
        qa_chain = create_qa_chain(retriever, llm)
        response = qa_chain.invoke({
            "question": query,
            "chat_history": history_input
        })
        
        answer = response.get("answer", "")
        source_docs = response.get("source_documents", [])
        
        return answer, source_docs
        
    except Exception as e:
        st.error(f"❌ Query processing failed: {e}")
        return f"❌ Error processing query: {e}", []

# UI Components

def render_authentication_sidebar():
    """Render authentication section in sidebar."""
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown("### 🔐 Authentication")
    
    if st.session_state.get('user'):
        st.sidebar.success(f"👤 Welcome, {st.session_state['user']}!")
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    else:
        auth_mode = st.sidebar.radio("Select Mode:", ["Login", "Signup"])
        
        with st.sidebar.form("auth_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submit_btn = st.form_submit_button(f"✅ {auth_mode}", use_container_width=True)
            
            if submit_btn and username and password:
                if auth_mode == "Signup":
                    success, message = signup_user(username, password)
                    if success:
                        st.session_state['user'] = username
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    success, message = login_user(username, password)
                    if success:
                        st.session_state['user'] = username
                        st.rerun()
                    else:
                        st.error(message)
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

def render_chat_management_sidebar():
    """Render chat management section in sidebar."""
    if not st.session_state.get('user'):
        return
    
    # st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown("### 💬 Chat Management")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🆕 New Chat", use_container_width=True):
            st.session_state.update({
                'chat_history': [],
                'summary_memory': "",
                'current_chat_id': None
            })
            st.rerun()
    
    with col2:
        if st.button("💾 Save Chat", use_container_width=True):
            if save_chat_to_db(st.session_state['user']):
                st.success("Chat saved!")
    
    # Search chats
    search_query = st.sidebar.text_input("🔍 Search chats:", placeholder="Search your conversations...")
    
    # Load and display chat history
    user_chats = load_user_chats(st.session_state['user'])
    
    if user_chats:
        st.sidebar.markdown("#### Recent Conversations")
        for chat in user_chats:
            chat_id = str(chat['_id'])
            recent_turns = chat.get('recent_turns', [])
            
            timestamp_data = chat.get('timestamp')
            
            if isinstance(timestamp_data, (int, float)):
                try:
                    timestamp_dt = datetime.datetime.fromtimestamp(timestamp_data)
                    timestamp = timestamp_dt.strftime("%m/%d %H:%M")
                except (ValueError, TypeError):
                    timestamp = "Invalid Date"
            elif isinstance(timestamp_data, datetime.datetime):
                timestamp = timestamp_data.strftime("%m/%d %H:%M")
            else:
                timestamp = "No Timestamp"

            if recent_turns:
                first_question = recent_turns[0][0]
                if search_query and search_query.lower() not in first_question.lower():
                    continue
                    
                display_name = (first_question[:30] + "...") if len(first_question) > 30 else first_question
                # timestamp = chat['timestamp'].strftime("%m/%d %H:%M")
                
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    if st.button(f"{display_name}", key=f"load_{chat_id}", use_container_width=True):
                        if load_specific_chat(chat_id):
                            st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"del_{chat_id}", help="Delete chat"):
                        if delete_chat(chat_id):
                            st.rerun()
                
                st.sidebar.caption(f"📅 {timestamp}")
                st.sidebar.divider()
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

def render_settings_sidebar():
    """Render settings section in sidebar."""
    if not st.session_state.get('user'):
        return
    
    # st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown("### ⚙️ Settings")
    
    # Collection selection
    collection_name = st.sidebar.selectbox(
        "📚 Document Collection:",
        list(COLLECTION_EMBED_MAP.keys()),
        index=list(COLLECTION_EMBED_MAP.keys()).index(st.session_state.get('collection_name', list(COLLECTION_EMBED_MAP.keys())[0])),
        help="Choose the document collection to search"
    )
    
    # Show collection description
    if collection_name in COLLECTION_EMBED_MAP:
        st.sidebar.caption(COLLECTION_EMBED_MAP[collection_name]["description"])
    
    # LLM selection
    llm_model = st.sidebar.selectbox(
        "🤖 AI Model:",
        list(GROQ_MODELS.keys()),
        index=list(GROQ_MODELS.keys()).index(st.session_state.get('llm_model', list(GROQ_MODELS.keys())[0])),
        format_func=lambda x: GROQ_MODELS[x],
        help="Choose the AI model for generating responses"
    )
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        k_value = st.slider("Top K Documents:", 1, 10, st.session_state.get('k_value', 5), help="Number of documents to retrieve")
        alpha_value = st.slider("Hybrid Search Alpha:", 0.0, 1.0, st.session_state.get('alpha_value', 0.5), help="0=keyword search, 1=semantic search")
        show_metadata = st.checkbox("📋 Show Sources", value=st.session_state.get('show_metadata', True))
    
    # Apply settings button
    if st.sidebar.button("✅ Apply Settings", use_container_width=True):
        st.session_state.update({
            'collection_name': collection_name,
            'llm_model': llm_model,
            'k_value': k_value,
            'alpha_value': alpha_value,
            'show_metadata': show_metadata,
            'retriever': build_retriever_for_collection(collection_name, k_value, alpha_value),
            'llm': build_llm(llm_model)
        })
        st.success("⚡ Settings applied!")
        time.sleep(1)
        st.rerun()
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

def render_chat_interface():
    """Render main chat interface."""
    # Header
    st.markdown('<div class="main-header">⚖️ Legal AI Assistant</div>', unsafe_allow_html=True)
    
    if not st.session_state.get('user'):
        st.info("👈 Please login or signup from the sidebar to start chatting")
        
        # Show demo information
        col1, col2, col3 = st.columns(3)
        with col1:
            # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📚 Collections", len(COLLECTION_EMBED_MAP))
            st.markdown("Legal document collections")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🤖 AI Models", len(GROQ_MODELS))
            st.markdown("Advanced language models")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🔍 Search Types", "2")
            st.markdown("Hybrid retrieval system")
            st.markdown('</div>', unsafe_allow_html=True)
        
        return
    
    # Initialize components if needed
    if not st.session_state.get('retriever'):
        st.session_state['retriever'] = build_retriever_for_collection(
            st.session_state.get('collection_name', list(COLLECTION_EMBED_MAP.keys())[0]),
            st.session_state.get('k_value', 5),
            st.session_state.get('alpha_value', 0.5)
        )
    
    if not st.session_state.get('llm'):
        st.session_state['llm'] = build_llm(
            st.session_state.get('llm_model', list(GROQ_MODELS.keys())[0])
        )
    
    # Display current settings
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📚 Collection: {st.session_state.get('collection_name', 'None')}")
    with col2:
        model_name = st.session_state.get('llm_model', 'None')
        display_name = GROQ_MODELS.get(model_name, model_name)
        st.info(f"🤖 Model: {display_name}")
    with col3:
        chat_count = len(st.session_state.get('chat_history', []))
        st.info(f"💬 Messages: {chat_count}")
    
    # Chat display area
    chat_container = st.container()
    
    # Display chat history
    if st.session_state.get('chat_history'):
        for i, (user_msg, assistant_msg) in enumerate(st.session_state['chat_history']):
            with chat_container:
                # User message
                # st.markdown(f'<div class="chat-message user-message">👤 <strong>You:</strong><br>{user_msg}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-message user-message">👤 <strong>You:</strong><br>{user_msg}</div>', unsafe_allow_html=True)

                
                # Assistant message
                st.markdown(f'🤖 <strong>Legal Assistant:</strong><br>{assistant_msg}</div>', unsafe_allow_html=True)
                
                st.markdown("---")
    
    # Chat input
    user_input = st.chat_input("💭 Ask me about legal documents...", key="chat_input")
    
    if user_input:
        # Add user message to history immediately for display
        st.session_state['chat_history'].append((user_input, ""))
        
        # Process query with loading indicator
        with st.spinner("🔍 Searching documents and generating response..."):
            answer, sources = process_query(user_input)
        
        # Update the last message with the response
        st.session_state['chat_history'][-1] = (user_input, answer)
        
        # Auto-save chat
        save_chat_to_db(st.session_state['user'])
        
        # Store sources for display
        st.session_state['last_sources'] = sources
        
        st.rerun()
    
    # Display sources for the last query
    if st.session_state.get('show_metadata') and st.session_state.get('last_sources'):
        st.markdown("### 📑 Sources")
        
        sources = st.session_state['last_sources']
        if sources:
            # Group sources by case title
            sources_by_case = {}
            for doc in sources:
                case_title = doc.metadata.get('case_title', 'Unknown Case')
                if case_title not in sources_by_case:
                    sources_by_case[case_title] = []
                sources_by_case[case_title].append(doc)
            
            # Display sources
            for case_title, case_docs in sources_by_case.items():
                with st.expander(f"📋**Case Title :** {case_title}", expanded=len(sources_by_case) == 1):
                    for doc in case_docs:
                        meta = doc.metadata
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            # st.markdown(f'<div class="source-box">', unsafe_allow_html=True)
                            st.markdown(f"**Court:** {meta.get('court', 'N/A')}")
                            st.markdown(f"**File:** {meta.get('file_name', 'N/A')}")
                            st.markdown(f"**Chunk:** {meta.get('chunk_index', 'N/A')}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            score = meta.get('score', 0)
                            st.metric("Relevance", f"{score:.2f}")

                        content = doc.page_content or ""
                        if len(content) > 500:
                            content = content[:500] + "..."
                        st.text_area("Content Preview:", content, height=100, disabled=True, key=f"content_{meta.get('chunk_index', 0)}")
                        
                        st.markdown("---")
        else:
            st.info("No sources available for the last query.")

def render_export_functionality():
    """Render export functionality."""
    if not st.session_state.get('user') or not st.session_state.get('chat_history'):
        return
    
    # st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown("### 📤 Export Chat")
    
    if st.sidebar.button("📄 Export as JSON", use_container_width=True):
        export_data = {
            'user': st.session_state['user'],
            'timestamp': datetime.datetime.now().isoformat(),
            'collection': st.session_state.get('collection_name'),
            'llm_model': st.session_state.get('llm_model'),
            'chat_history': st.session_state['chat_history'],
            'summary_memory': st.session_state.get('summary_memory', ''),
            'settings': {
                'k': st.session_state.get('k_value', 5),
                'alpha': st.session_state.get('alpha_value', 0.5)
            }
        }
        
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        st.sidebar.download_button(
            "⬇️ Download JSON",
            data=json_str,
            file_name=f"legal_chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    if st.sidebar.button("📊 Export as CSV", use_container_width=True):
        if st.session_state['chat_history']:
            df = pd.DataFrame(st.session_state['chat_history'], columns=['Question', 'Answer'])
            df['Timestamp'] = datetime.datetime.now()
            df['User'] = st.session_state['user']
            df['Collection'] = st.session_state.get('collection_name')
            df['Model'] = st.session_state.get('llm_model')
            
            csv_str = df.to_csv(index=False)
            st.sidebar.download_button(
                "⬇️ Download CSV",
                data=csv_str,
                file_name=f"legal_chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

def render_statistics():
    return
#     """Render usage statistics."""
#     if not st.session_state.get('user'):
#         return
    
#     st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
#     st.sidebar.markdown("### 📊 Statistics")
    
#     user_chats = load_user_chats(st.session_state['user'], limit=100)
#     total_chats = len(user_chats)
#     total_messages = sum(len(chat.get('recent_turns', [])) for chat in user_chats)
    
#     col1, col2 = st.sidebar.columns(2)
#     with col1:
#         st.metric("💬 Total Chats", total_chats)
#     with col2:
#         st.metric("📝 Total Messages", total_messages)

#     if user_chats:
#         # Most used collection
#         collections = [chat.get('collection', 'Unknown') for chat in user_chats]
#         most_used = max(set(collections), key=collections.count)
#         st.sidebar.markdown(f"**Most Used Collection:** {most_used}")

#         # Recent activity
#         recent_chat = user_chats[0] if user_chats else None
#         if recent_chat and 'timestamp' in recent_chat:
#             timestamp_data = recent_chat['timestamp']

#             if isinstance(timestamp_data, (float, int)):
#                 last_activity_dt = datetime.datetime.fromtimestamp(timestamp_data)
#             else: 
#                 last_activity_dt = timestamp_data

#             last_activity = last_activity_dt.strftime("%Y-%m-%d %H:%M")
#             st.sidebar.markdown(f"**Last Activity:** {last_activity}")

#     st.sidebar.markdown('</div>', unsafe_allow_html=True)

def render_help_section():
    """Render help and tips section."""
    # st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown("### ❓ Help & Tips")
    
    with st.sidebar.expander("📖 How to Use"):
        st.markdown("""
        **Getting Started:**
        1. Login or create an account
        2. Select a document collection
        3. Choose an AI model
        4. Start asking legal questions!
        
        **Tips for Better Results:**
        - Be specific in your questions
        - Reference case names or legal concepts
        - Use legal terminology when appropriate
        - Ask follow-up questions for clarification
        """)
    
    with st.sidebar.expander("⚙️ Settings Guide"):
        st.markdown("""
        **Collection Types:**
        - **InLegalBERT**: Best for specialized legal queries
        - **Legal BERT**: Good for case analysis
        - **All-mpnet**: General legal questions
        
        **Search Settings:**
        - **Top K**: Number of relevant documents to find
        - **Alpha**: Balance between keyword and semantic search
        - **Lower Alpha**: More keyword matching
        - **Higher Alpha**: More semantic understanding
        """)
    
    with st.sidebar.expander("🚨 Important Notes"):
        st.markdown("""
        **Disclaimers:**
        - This tool provides information, not legal advice
        - Always consult qualified legal professionals
        - Verify information from official sources
        - Keep sensitive information confidential
        """)
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Main Application

def main():
    """Main application function."""
    init_session_state()
    
    if not weaviate_client:
        st.error("❌ Weaviate connection failed. Please check your configuration.")
        return
    
    if not GROQ_API_KEY:
        st.error("❌ GROQ API key not found. Please check your environment variables.")
        return
    
    if not mongo_client:
        st.warning("⚠️ MongoDB connection failed. Authentication and chat history will not work.")
    
    render_authentication_sidebar()
    
    if st.session_state.get('user'):
        render_chat_management_sidebar()
        render_settings_sidebar()
        render_export_functionality()
        render_statistics()
    
    render_help_section()
    
    render_chat_interface()
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>⚖️ <strong>Legal AI Assistant</strong> | Powered by Weaviate, Groq, and Streamlit</p>
        <p><em>This tool provides information only and is not a substitute for professional legal advice.</em></p>
    </div>
    """, unsafe_allow_html=True)

# Application Entry Point

if __name__ == "__main__":
    main()