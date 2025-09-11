import streamlit as st

import time
import os
import re
import datetime
import uuid
import json
import hashlib
from bson.objectid import ObjectId
from dotenv import load_dotenv

import weaviate
from weaviate.auth import Auth

from pydantic import Field, PrivateAttr
from typing import List

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Page config
st.set_page_config(
    page_title="Legal RAG Assistant", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f4e79;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #ff4b4b;
    }
    .assistant-message {
        background-color: #e8f4fd;
        border-left-color: #1f4e79;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f4e79;
        margin-bottom: 1rem;
    }
    .settings-expander {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
    "meta-llama/llama-4-maverick-17b-128e-instruct": "Meta Llama 17b (Balanced)",
    "gemma2-9b-it": "Gemma 2 9B (Efficient)",
    "openai/gpt-oss-120b": "OpenAI 120B",
    "openai/gpt-oss-20b": "OpenAI 20B"
}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
WEA_URL = os.environ.get("WEAVIATE_URL")
WEA_KEY = os.environ.get("WEAVIATE_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")

@st.cache_resource
def init_mongodb():
    """Initialize MongoDB connection"""
    try:
        client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
        db = client["Agentic_RAG"]
        client.admin.command('ping')
        return db
    except Exception as e:
        st.error(f"MongoDB connection failed: {e}")
        return None

@st.cache_resource
def init_weaviate():
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

class WeaviateHybridRetriever(BaseRetriever):
    """Custom hybrid retriever for Weaviate using both semantic and keyword search"""

    client: weaviate.WeaviateClient = Field(..., description="Weaviate client instance.")
    collection_name: str = Field(..., description="Weaviate collection name.")
    embedding_model_name: str = Field(..., description="HuggingFace embedding model.")
    alpha: float = Field(0.5, description="Hybrid search alpha (0=keyword, 1=vector).")
    k: int = Field(5, description="Number of documents to retrieve.")

    _embeddings: HuggingFaceEmbeddings = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        """Retrieve documents using hybrid search"""
        try:
            collection = self.client.collections.get(self.collection_name)
            query_vector = self._embeddings.embed_query(query)

            results = collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=self.alpha,
                limit=self.k,
                return_properties=["text", "case_title", "court", "file_name", "chunk_index"],
                return_metadata=weaviate.classes.query.MetadataQuery(score=True),
            )

            documents = []
            for obj in results.objects:
                props = obj.properties
                score = obj.metadata.score if obj.metadata and obj.metadata.score is not None else 0.0
                documents.append(
                    Document(
                        page_content=props.get("text", ""),
                        metadata={
                            "case_title": props.get("case_title", ""),
                            "court": props.get("court", ""),
                            "file_name": props.get("file_name", ""),
                            "chunk_index": props.get("chunk_index", 0),
                            "score": score,
                        },
                    )
                )
            return documents

        except Exception as e:
            st.error(f"❌ Error in retrieval: {e}")
            return []

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)

def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(db, username: str, email: str, password: str) -> bool:
    """Create new user"""
    try:
        users_collection = db["users"]
        
        # Check if user already exists
        if users_collection.find_one({"$or": [{"username": username}, {"email": email}]}):
            return False
        
        user_doc = {
            "username": username,
            "email": email,
            "password": hash_password(password),
            "created_at": datetime.datetime.now(),
            "last_login": datetime.datetime.now()
        }
        
        users_collection.insert_one(user_doc)
        return True
    except Exception as e:
        st.error(f"Error creating user: {e}")
        return False

def authenticate_user(db, username: str, password: str) -> dict:
    """Authenticate user"""
    try:
        users_collection = db["users"]
        user = users_collection.find_one({"username": username})
        
        if user and user["password"] == hash_password(password):
            # Update last login
            users_collection.update_one(
                {"username": username},
                {"$set": {"last_login": datetime.datetime.now()}}
            )
            return user
        return None
    except Exception as e:
        st.error(f"Error authenticating user: {e}")
        return None

def save_chat_message(db, user_id, chat_id, message_type, content, metadata=None):
    """Save chat message to MongoDB"""
    try:
        chats_collection = db["chats"]
        
        message = {
            "user_id": user_id,
            "chat_id": chat_id,
            "message_type": message_type,
            "content": content,
            "timestamp": datetime.datetime.now(),
            "metadata": metadata or {}
        }
        
        chats_collection.insert_one(message)
    except Exception as e:
        st.error(f"Error saving message: {e}")

def get_user_chats(db, user_id):
    """Get all chats for a user"""
    try:
        chats_collection = db["chats"]
        
        # Get unique chat_ids for the user with latest message
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$sort": {"timestamp": -1}},
            {"$group": {
                "_id": "$chat_id",
                "latest_message": {"$first": "$content"},
                "latest_timestamp": {"$first": "$timestamp"}
            }},
            {"$sort": {"latest_timestamp": -1}}
        ]
        
        return list(chats_collection.aggregate(pipeline))
    except Exception as e:
        st.error(f"Error getting chats: {e}")
        return []

def get_chat_messages(db, chat_id):
    """Get all messages for a specific chat"""
    try:
        chats_collection = db["chats"]
        return list(chats_collection.find({"chat_id": chat_id}).sort("timestamp", 1))
    except Exception as e:
        st.error(f"Error getting chat messages: {e}")
        return []

# Utility functions
def is_greeting_or_casual(query: str) -> bool:
    """Check if query is a greeting or casual conversation"""
    casual_patterns = [
        r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
        r'\b(how are you|what\'s up|thanks|thank you|bye|goodbye)\b',
        r'\b(please|sorry|excuse me)\b$',
        r'^\s*(hi|hello|hey|thanks|thank you|bye)\s*$'
    ]
    
    query_lower = query.lower().strip()
    return any(re.search(pattern, query_lower, re.IGNORECASE) for pattern in casual_patterns)

def create_smart_retrieval_function(retriever, show_metadata=False):
    """Create smart retrieval function with metadata option"""
    def smart_retrieval_function(query: str) -> str:
        if is_greeting_or_casual(query):
            return "CASUAL_QUERY_NO_SEARCH_NEEDED"
        
        try:
            documents = retriever.invoke(query)
            if not documents:
                return "No relevant documents found for the query."
            
            formatted_results = []
            for i, doc in enumerate(documents, 1):
                metadata = doc.metadata
                result = f"""
Document {i}:
Case Title: {metadata.get('case_title', 'N/A')}
Court: {metadata.get('court', 'N/A')}
File: {metadata.get('file_name', 'N/A')}
Score: {metadata.get('score', 'N/A'):.3f}
Content: {doc.page_content[:400]}{'...' if len(doc.page_content) > 400 else ''}
---"""
                formatted_results.append(result)
                
                # Store metadata in session state for display
                if show_metadata and 'last_retrieval_metadata' not in st.session_state:
                    st.session_state.last_retrieval_metadata = []
                if show_metadata:
                    st.session_state.last_retrieval_metadata.append(metadata)
            
            return "\n".join(formatted_results)
        
        except Exception as e:
            return f"Error during retrieval: {str(e)}"
    
    return smart_retrieval_function

# Enhanced system prompt
ENHANCED_SYSTEM_PROMPT = """
You are an AI Legal Research Assistant specializing in legal document analysis and case law research.

CORE CAPABILITIES:
- Legal document analysis and interpretation
- Case law research and precedent identification
- Legal concept explanation and clarification
- Statutory interpretation and analysis

RESPONSE PROTOCOL:

For CASUAL/GREETING queries (hi, hello, thanks, etc.):
- Respond professionally and briefly (under 20 words)
- DO NOT use the document retrieval tool
- Example: "Hello! I'm here to help with your legal research needs."

For LEGAL RESEARCH queries:
1. ALWAYS use the LegalDocsRetriever tool first
2. Analyze retrieved documents thoroughly
3. Provide structured responses with:
   - Direct answer based on retrieved documents
   - Supporting evidence with source citations
   - Relevant case references when available
   - Legal disclaimer

RESPONSE FORMAT:
- Lead with a clear, concise answer
- Support with specific document references
- Include relevance scores when significant
- End with appropriate legal disclaimer

IMPORTANT CONSTRAINTS:
- Use ONLY information from retrieved legal documents
- If no relevant documents found, state clearly: "I cannot find relevant information in the available legal documents."
- Never provide legal advice - only document-based information
- Always include disclaimer: "This information is for research purposes only and does not constitute legal advice."

Maintain professional tone throughout all interactions.
"""

def main():
    # Initialize connections
    db = init_mongodb()
    weaviate_client = init_weaviate()
    
    if db is None:
        st.error("⚠️ Database connection failed. Please check your MongoDB configuration.")
        return
    
    if not weaviate_client:
        st.error("⚠️ Weaviate connection failed. Please check your configuration.")
        return

    # Authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        show_auth_page(db)
        return
    
    # Main app interface
    show_main_app(db, weaviate_client)

def show_auth_page(db):
    """Show authentication page"""
    st.markdown('<h1 class="main-header">⚖️ Legal RAG Assistant</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_button = st.form_submit_button("Login")
                
                if login_button and username and password:
                    user = authenticate_user(db, username, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
        
        with tab2:
            st.subheader("Register")
            with st.form("register_form"):
                new_username = st.text_input("Username")
                email = st.text_input("Email")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                register_button = st.form_submit_button("Register")
                
                if register_button:
                    if not all([new_username, email, new_password, confirm_password]):
                        st.error("Please fill in all fields")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        if create_user(db, new_username, email, new_password):
                            st.success("✅ Registration successful! Please login.")
                        else:
                            st.error("❌ Username or email already exists")

def show_main_app(db, weaviate_client):
    """Show main application interface"""
    # Header
    st.markdown('<h1 class="main-header">⚖️ Legal RAG Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f'<p class="sidebar-header">Welcome, {st.session_state.user["username"]}!</p>', unsafe_allow_html=True)
        
        if st.button("Logout", type="secondary"):
            st.session_state.clear()
            st.rerun()
        
        st.divider()
        
        # Model Selection
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
        
        # Chat History
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
        
        st.divider()
        
        # Help & Tips
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
    
    # Main chat interface
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = str(uuid.uuid4())
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
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
        
        # Create LLM
        llm = ChatGroq(
            model=selected_model,
            groq_api_key=GROQ_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Create memory LLM (using Gemini for summarization)
        memory_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            max_output_tokens=500
        )
        
        # Create memory
        memory = ConversationSummaryBufferMemory(
            llm=memory_llm,
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=1000,
            human_prefix="User",
            ai_prefix="Assistant"
        )
        
        # Create tool
        retrieval_function = create_smart_retrieval_function(retriever, show_metadata)
        retrieval_tool = Tool(
            name="LegalDocsRetriever",
            func=retrieval_function,
            description="Retrieves relevant legal documents and case information. Use for legal questions only."
        )
        
        # Create agent
        agent = initialize_agent(
            tools=[retrieval_tool],
            llm=llm,
            agent="zero-shot-react-description",
            verbose=False,
            handle_parsing_errors=True,
            memory=memory,
            agent_kwargs={"prefix": ENHANCED_SYSTEM_PROMPT}
            # max_iterations=5
        )
        
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return
    
    # Display messages
    for message in st.session_state.messages:
        if message["message_type"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Show metadata if enabled and available
                if show_metadata and "metadata" in message and message["metadata"]:
                    with st.expander("📊 Document Metadata"):
                        # for i, meta in enumerate(message["metadata"], 1):
                        #     st.json({
                        #         f"Document {i}": {
                        #             "Case Title": meta.get("case_title", "N/A"),
                        #             "Court": meta.get("court", "N/A"),
                        #             "File": meta.get("file_name", "N/A"),
                        #             "Score": f"{meta.get('score', 0):.3f}",
                        #             "Chunk Index": meta.get("chunk_index", 0)
                        #         }
                        #     })
                        for i, meta in enumerate(message["metadata"], 1):
                            # Ensure metadata is a dict
                            if isinstance(meta, str):
                                try:
                                    meta = json.loads(meta)  # convert JSON string → dict
                                except json.JSONDecodeError:
                                    meta = {}  # fallback if it's not valid JSON
                        
                            st.json({
                                f"Document {i}": {
                                    "Case Title": meta.get("case_title", "N/A"),
                                    "Court": meta.get("court", "N/A"),
                                    "File": meta.get("file_name", "N/A"),
                                    "Score": f"{meta.get('score', 0):.3f}",
                                    "Chunk Index": meta.get("chunk_index", 0)
                                }
                            })
    
    # Chat input
    if query := st.chat_input("Ask me about legal documents and cases..."):
        # Add user message
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
                    
                    # Show metadata if enabled
                    if show_metadata and metadata:
                        with st.expander("📊 Document Metadata"):
                            for i, meta in enumerate(metadata, 1):
                                st.json({
                                    f"Document {i}": {
                                        "Case Title": meta.get("case_title", "N/A"),
                                        "Court": meta.get("court", "N/A"),
                                        "File": meta.get("file_name", "N/A"),
                                        "Score": f"{meta.get('score', 0):.3f}",
                                        "Chunk Index": meta.get("chunk_index", 0)
                                    }
                                })
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "message_type": "assistant",
                        "content": response_text,
                        "timestamp": datetime.datetime.now(),
                        "metadata": metadata
                    })
                    
                    # Save assistant message to DB
                    save_chat_message(
                        db,
                        st.session_state.user["_id"],
                        st.session_state.current_chat_id,
                        "assistant",
                        response_text,
                        {"retrieval_metadata": metadata}
                    )
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    
                    st.session_state.messages.append({
                        "message_type": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.datetime.now()
                    })

if __name__ == "__main__":
    main()