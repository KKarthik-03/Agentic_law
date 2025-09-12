# Agentic Legal Assistant  

An **AI-powered legal assistant** built with **Streamlit** that leverages **agentic reasoning** and **retrieval-augmented generation (RAG)** to answer legal queries. The system integrates **multiple domain-specific embeddings** and advanced **LLMs (Groq + Gemini 2.0 Flash)** to provide accurate, contextual, and explainable responses.  

---

## 🚀 Features  

- ⚖️ **Legal-Specific Knowledge**: Uses three specialized embedding collections:  
  - **InLegalBERT** – Indian legal documents *(law-ai/InLegalBERT)*  
  - **LegalBERT** – General legal text corpora *(nlpaueb/legal-bert-base-uncased)*  
  - **All-MiniLM-MPNet v2** – General semantic similarity *(sentence-transformers/all-mpnet-base-v2)*  

- 🤖 **Agentic RAG Pipeline**: Combines retrieval + reasoning for accurate responses.  

- 🧠 **Memory with Summarization**: Uses **Gemini 2.0 Flash** to summarize old chat history into compact memory.  

- ⚡ **Groq-Powered LLMs**: Ultra-fast inference for legal query answering.  

- 💡 **User-Friendly UI**: Interactive **Streamlit app** with built-in **help and tips**.  

- 💬 **Chat Sessions**:  
  - Persistent chat history stored in **MongoDB**.  
  - Multiple sessions with **unique titles per user**.  

---

## 🏗️ Architecture  

1. **Document Storage**: Legal documents embedded and stored across 3 collections (InLegalBERT, LegalBERT, MPNet).  
2. **Retriever**: Hybrid retrieval mechanism queries relevant passages from all collections.  
3. **Agentic Orchestration**: LangChain agents decide which tools/collections to use.  
4. **LLM Layer**:  
   - **Groq Models** – reasoning & query answering.  
   - **Gemini 2.0 Flash** – summarizing chat history into memory.  
5. **UI Layer**: Streamlit app with guidance & tips for users.  

---

## 📦 Installation  

Clone the repo and install dependencies:  

```bash
git clone <https://github.com/KKarthik-03/Agentic_law>
cd legal-case-assistant
pip install -r requirements.txt
```

Environment Setup

Create a .env file with your API keys:
```
# Groq API
GROQ_API_KEY=your_groq_api_key

# Google Gemini API
GOOGLE_API_KEY=your_google_api_key

# MongoDB (for storing chat / metadata)
MONGODB_URI=your_mongodb_uri
```

▶️ Running the App
```
streamlit run app.py
```

🖥️ Usage

- Upload or query legal documents.
- Ask natural-language legal questions.
- The agent retrieves from embeddings + memory.
- Get structured answers with case deatils.
- Use built-in help & tips for guidance ( under the Chat History ).

📂 Case Documents Used

- Application of the International Convention for the Suppression of the Financing of Terrorism and of the International Convention on the Elimination of All Forms of Racial Discrimination (Ukraine v. Russian Federation)
- Nirmal Singh v. Canada
- Mason v. Canada (Citizenship and Immigration)
- Canada (Public Safety and Emergency Preparedness) v. Chhina, 2019 SCC 29
- Legality of the Use by a State of Nuclear Weapons in Armed Conflict
- Fisheries Jurisdiction (Spain v. Canada)
- Canadian Council for Refugees v. Canada (Citizenship and Immigration)

🛠️ Tech Stack

- Frontend & UI: Streamlit
- Database: MongoDB (Users, Sessions, Chat History)
- Framework : Langchain
- Vector Store: Weaviate
- Authentication: bcrypt
- LLMs: Groq models & Gemini 2.0 Flash

Live Demo :
- HuggingFace Space : [Click Here](https://kodamkarthik281-ai-legal-case-assistant.hf.space)
