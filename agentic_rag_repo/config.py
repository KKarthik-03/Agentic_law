#  config.py --> All configuration constants and environment variables

import os
import weaviate
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WEA_URL = os.getenv("WEAVIATE_URL")
WEA_KEY = os.getenv("WEAVIATE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

import os
from dotenv import load_dotenv

def init_weaviate():
    load_dotenv()
    url = os.getenv("WEA_URL")
    key = os.getenv("WEA_KEY")

    if not url or not key:
        # print("❌ Missing WEA_URL or WEA_KEY in .env")
        return None

    try:
        client = weaviate.Client(
            url=url,
            auth_client_secret=weaviate.AuthApiKey(key)
        )
        if client.is_ready():
            print("✅ Weaviate client initialized successfully")
            return client
        else:
            print("❌ Weaviate client not ready")
            return None
    except Exception as e:
        print(f"❌ Error initializing Weaviate: {e}")
        return None
    
init_weaviate()

# Collection and Model Configurations
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
    "meta-llama/llama-4-maverick-17b-128e-instruct": "Meta Llama 17b (Balanced)",
    "gemma2-9b-it": "Gemma 2 9B (Efficient)",
    "llama-3.3-70b-versatile": "Llama 3.3 70B (Recommended)",
    "llama-3.1-8b-instant": "Llama 3.1 8B (Fast)"
}

# System Prompt
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
- you can provide the legal advice - but end with a disclaimer.
- Always include disclaimer: "This information is for research purposes only and does not constitute legal advice."

Maintain professional tone throughout all interactions.
"""