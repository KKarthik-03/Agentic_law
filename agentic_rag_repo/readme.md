# Legal RAG Assistant

An AI-powered legal research assistant that combines Retrieval-Augmented Generation (RAG) with agentic capabilities for legal document analysis and case law research.

## Features

- 🔐 **User Authentication**: Secure login/registration system with MongoDB
- 📚 **Legal Document Retrieval**: Hybrid search across legal document collections
- 🤖 **Multiple LLM Support**: Integration with Groq models and Google Gemini
- 💬 **Chat Interface**: Persistent chat history with user sessions
- ⚙️ **Configurable Settings**: Adjustable retrieval and generation parameters
- 📊 **Metadata Display**: View document relevance scores and sources

## Architecture

The application is organized into modular components:

```
├── app.py                 # Main Streamlit entry point
├── main_app.py           # Core application logic
├── config.py             # Configuration and constants
├── database.py           # MongoDB operations
├── weaviate_client.py    # Weaviate connection and retriever
├── rag_utils.py          # RAG utilities and agent creation
├── auth_ui.py            # Authentication UI components
├── chat_ui.py            # Chat interface components
├── styles.py             # UI styling and CSS
├── requirements.txt      # Python dependencies
└── .env.example          # Environment variables template
```

## Setup

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/KKarthik-03/Agentic_law
cd legal-rag-assistant
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Fill in your API keys and connection strings:

- `GROQ_API_KEY`: Your Groq API key
- `GOOGLE_API_KEY`: Your Google AI API key  
- `WEAVIATE_URL`: Your Weaviate cluster URL
- `WEAVIATE_API_KEY`: Your Weaviate API key
- `MONGO_URI`: Your MongoDB connection string

### 3. Database Setup

Ensure your MongoDB instance is running and accessible. The application will automatically create the required collections (`users` and `chats`).

### 4. Weaviate Setup

Your Weaviate instance should have legal document collections with the following properties:
- `text`: Document content
- `case_title`: Case title
- `court`: Court name
- `file_name`: Source file name
- `chunk_index`: Document chunk index

### 5. Run the Application

```bash
streamlit run app.py
```

## Usage

### Authentication
1. Register a new account or login with existing credentials
2. User data is securely stored in MongoDB with hashed passwords

### Legal Research
1. Select your preferred LLM model and knowledge base
2. Adjust retrieval settings (hybrid search alpha, top-k results)
3. Configure generation parameters (temperature, max tokens)
4. Ask legal questions in natural language

### Chat Management
- Start new conversations or continue previous chats
- All chat history is preserved per user
- View document metadata and relevance scores

## Supported Models

### LLM Models (via Groq)
- Llama 3.3 70B (Recommended)
- Llama 3.1 8B (Fast)
- Meta Llama 17B (Balanced)
- Gemma 2 9B (Efficient)

### Knowledge Bases
- **InLegalBERT_Chunks**: Specialized legal domain model
- **Legal_Bert_Chunks**: Legal BERT for case analysis
- **All_mpnet_Chunks**: General purpose sentence transformer

## Configuration Options

### Retrieval Settings
- **Alpha**: Controls hybrid search balance (0=keyword, 1=semantic)
- **Top K**: Number of documents to retrieve (1-10)
- **Show Metadata**: Display document sources and scores

### Generation Settings
- **Temperature**: Response creativity (0.0-1.0)
- **Max Tokens**: Maximum response length (500-4000)

## API Integration

The system integrates with:
- **Groq**: Fast LLM inference
- **Google Gemini**: Conversation summarization
- **Weaviate**: Vector database for document retrieval
- **MongoDB**: User data and chat persistence
- **Hugging Face**: Embedding models

## Security Features

- Password hashing with SHA-256
- Session-based authentication
- User data isolation
- Environment variable protection


## Troubleshooting

### Common Issues

**MongoDB Connection Failed**
- Verify `MONGO_URI` in `.env`
- Check network connectivity
- Ensure MongoDB instance is running

**Weaviate Connection Failed**  
- Verify `WEAVIATE_URL` and `WEAVIATE_API_KEY`
- Check cluster status
- Ensure collections exist with proper schema

**API Key Errors**
- Verify all API keys in `.env`
- Check key validity and quotas
- Ensure proper permissions