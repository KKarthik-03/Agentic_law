# rag_utils.py  --> RAG utilities, agent creation, and helper functions (Updated for RAGAS)

import re
import streamlit as st
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent
from langchain.memory import ConversationSummaryBufferMemory

from config import GROQ_API_KEY, GOOGLE_API_KEY, ENHANCED_SYSTEM_PROMPT

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
    """Create smart retrieval function with metadata option and context capture for RAGAS"""
    def smart_retrieval_function(query: str) -> str:
        if is_greeting_or_casual(query):
            return "CASUAL_QUERY_NO_SEARCH_NEEDED"
        
        try:
            documents = retriever.invoke(query)
            if not documents:
                return "No relevant documents found for the query."
            
            # Initialize session state lists for RAGAS evaluation
            if 'last_retrieval_contexts' not in st.session_state:
                st.session_state.last_retrieval_contexts = []
            if show_metadata and 'last_retrieval_metadata' not in st.session_state:
                st.session_state.last_retrieval_metadata = []

            # Clear previous data
            st.session_state.last_retrieval_contexts.clear()
            if show_metadata:
                st.session_state.last_retrieval_metadata.clear()

            formatted_results = []
            for i, doc in enumerate(documents, 1):
                metadata = doc.metadata or {}
                score = float(metadata.get('score', 0.0) or 0.0)
                
                # Store full context for RAGAS evaluation
                st.session_state.last_retrieval_contexts.append(doc.page_content)
                
                result = (
                    f"Document {i}:\n"
                    f"Case Title: {metadata.get('case_title', 'N/A')}\n"
                    f"Court: {metadata.get('court', 'N/A')}\n"
                    f"File: {metadata.get('file_name', 'N/A')}\n"
                    f"Score: {score:.3f}\n"
                    f"Content: {doc.page_content[:400]}{'...' if len(doc.page_content) > 400 else ''}\n---"
                )
                formatted_results.append(result)
                
                if show_metadata:
                    # Store metadata with full content for RAGAS
                    metadata_with_content = metadata.copy()
                    metadata_with_content['content'] = doc.page_content
                    st.session_state.last_retrieval_metadata.append(metadata_with_content)

            return "\n".join(formatted_results)

        except Exception as e:
            return f"Error during retrieval: {str(e)}"

    return smart_retrieval_function

def create_rag_agent(retriever, selected_model, temperature, max_tokens, alpha, top_k, show_metadata=False):
    """Create and configure the RAG agent with all components"""
    try:
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
        )
        
        return agent
        
    except Exception as e:
        st.error(f"Error creating RAG agent: {e}")
        return None


def run_agent_query(agent, query: str, retriever=None, show_metadata=False):
    """
    Run a query through the agentic RAG system and return structured results.
    Enhanced to support RAGAS evaluation with proper context capture.
    """
    try:
        # Run agent
        response = agent.invoke({"input": query})
        final_answer = response["output"] if isinstance(response, dict) else str(response)

        # Get contexts from session state (populated by retrieval function)
        retrieved_docs = st.session_state.get('last_retrieval_contexts', [])
        retrieved_metadata = st.session_state.get('last_retrieval_metadata', [])

        # Fallback: direct retrieval if contexts not found in session state
        if not retrieved_docs and retriever:
            docs = retriever.invoke(query)
            for d in docs:
                retrieved_docs.append(d.page_content)
                retrieved_metadata.append(d.metadata)

        return {
            "query": query,
            "generated_answer": final_answer,
            "contexts": retrieved_docs,
            "contexts_metadata": retrieved_metadata,
        }

    except Exception as e:
        return {
            "query": query,
            "generated_answer": f"Error: {str(e)}",
            "contexts": [],
            "contexts_metadata": [],
        }