# 🤖 DocuBot - Intelligent Document Analysis Assistant

An advanced Retrieval-Augmented Generation (RAG) chatbot that analyzes uploaded documents and provides intelligent, context-aware responses. DocuBot combines semantic search, document re-ranking, and large language models to deliver accurate, sourced answers.

## ✨ Features

- **📄 Multi-Document Analysis** - Upload and analyze multiple PDF documents simultaneously
- **🔍 Intelligent Retrieval** - Two-stage retrieval pipeline with FAISS vector search and CrossEncoder re-ranking
- **💬 Context-Aware Responses** - Answers grounded in your documents with proper citations
- **📊 Data Visualization** - Generate interactive Chart.js visualizations in the web interface
- **📑 PDF Report Generation** - Export analysis and discussions as formatted PDF reports
- **💾 Persistent Memory** - Maintains conversation history across sessions
- **🌐 Dual Interface** - Both CLI and web-based (Streamlit) interfaces
- **⚡ Optimized Performance** - Normalized embeddings and efficient document chunking
- **🔐 Privacy-First** - All processing done locally with your own API keys

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                     DocuBot System                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌──────────────────────┐          │
│  │  Documents   │         │  API Keys (.env)     │          │
│  │  (PDFs)      │         │  - OPENAI_API_KEY    │          │
│  └──────┬───────┘         │  - GOOGLE_API_KEY    │          │
│         │                 └──────────────────────┘          │
│         ▼                                                    │
│  ┌─────────────────────────┐                               │
│  │  Ingest Pipeline        │                               │
│  │  - PDF Loading          │                               │
│  │  - Text Splitting       │                               │
│  │  - Embedding            │                               │
│  └──────────┬──────────────┘                               │
│             ▼                                               │
│  ┌─────────────────────────┐                               │
│  │  FAISS Vector Store     │                               │
│  │  (Local Embeddings)     │                               │
│  └──────────┬──────────────┘                               │
│             │                                              │
│  ┌──────────┴────────────────────────┐                   │
│  │                                    │                   │
│  ▼                                    ▼                   │
│  ┌─────────────────────────┐  ┌─────────────────────┐   │
│  │ Retrieval Pipeline      │  │  LLM Selection      │   │
│  │ 1. Coarse Search (k=10) │  │  - GPT-4.1 (Primary)│   │
│  │ 2. CrossEncoder Rerank  │  │  - Gemini (Fallback)│   │
│  │ 3. Top-4 Documents      │  └─────────────────────┘   │
│  └──────────┬──────────────┘                             │
│             │                                            │
│             ▼                                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Response Generation                            │   │
│  │  - Build System Prompt with Context             │   │
│  │  - Add Conversation History                     │   │
│  │  - LLM Inference                                │   │
│  │  - Tool Dispatch (PDF Generation, etc.)         │   │
│  └──────────┬────────────────────────────────────┘    │
│             │                                         │
│  ┌──────────┴──────────────┐                        │
│  │                         │                        │
│  ▼                         ▼                        │
│  ┌────────────────┐  ┌────────────────┐           │
│  │  CLI Output    │  │  Web Interface │           │
│  │  (chatbot.py)  │  │  (Streamlit)   │           │
│  └────────────────┘  └────────────────┘           │
│                                                    │
└────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | OpenAI GPT-4.1 (primary), Google Gemini 2.5 (fallback) |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector DB** | FAISS (local) |
| **Re-ranking** | CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| **Document Loading** | PyPDF |
| **Framework** | LangChain |
| **Web UI** | Streamlit |
| **PDF Generation** | ReportLab |

## 📋 Prerequisites

- Python 3.12+
- pip or conda
- API Keys:
  - OpenAI API key (for GPT-4.1) - [Get here](https://platform.openai.com/api-keys)
  - Google Gemini API key (optional, fallback) - [Get here](https://aistudio.google.com/app/apikey)

## 🚀 Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd chatbot-rag
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create a `.env` file in the project root:

```env
OPENAI_API_KEY=xx-xxxx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GOOGLE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**⚠️ Important**: Never commit `.env` file to version control!

### 5. Prepare Documents
Create an `input/` folder and add PDF documents:
```bash
mkdir input
# Copy your PDF files to input/
```

### 6. Ingest Documents
```bash
python ingest.py
```

This creates a `vectordb/` folder with FAISS embeddings.

## 💻 Usage

### Option 1: CLI Interface

#### Start the Chatbot
```bash
python chatbot.py
```

#### Example Conversation
```
🤖 DocuBot: Hello! I've analyzed your documents. How can I help? (Type 'exit' to stop)

You: What are the main financial highlights?

📄 Retrieved 4 relevant documents (3666 chars)

🤖 DocuBot: The main financial highlights from the Integrated Annual Report 2024-25 are:

1. **Revenue from Operations (Consolidated)**: ₹2,40,893 crore
2. **Profit for the Year**: ₹46,099 crore
3. **Earnings Per Share**: ₹125.88
...

You: Save this as a PDF

📄 DocuBot: Generating your PDF report...

🤖 DocuBot: ✅ Report successfully saved!

PDF created successfully at: /Users/pault/Documents/internship/regan/chatbot-rag/outputs/Summary_20260310_111213.pdf

You: exit
```

### Option 2: Web Interface (Streamlit)

#### Start the Web App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

#### Features
1. **📁 Sidebar - Document Management**
   - Upload multiple PDF files
   - Click "Ingest Documents" to process
   - Clear chat history button

2. **💬 Main Interface - Chat**
   - Type questions about documents
   - View responses with citations
   - See retrieved document count

3. **📊 Visualizations**
   - Automatic Chart.js rendering
   - Interactive charts for data analysis

4. **📄 PDF Export**
   - Generate reports from chat
   - Downloaded to `outputs/` folder

## 📁 Project Structure

```
chatbot-rag/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env                      # API keys (⚠️ don't commit)
│
├── 📄 Core Files
├── ingest.py                 # Document ingestion pipeline
├── chatbot.py                # CLI chatbot interface
├── app.py                    # Streamlit web interface
│
├── 🧠 Modular Components
├── utils.py                  # Utility functions (embeddings, PDF loading)
├── retrieval.py              # RAG retrieval pipeline (FAISS + reranking)
├── memory.py                 # Chat history persistence
├── tools.py                  # Tool definitions (PDF generation)
├── prompts.py                # System prompts and instructions
│
├── 📊 Data Directories
├── input/                    # Your PDF documents (created on first run)
├── outputs/                  # Generated PDF reports
├── vectordb/                 # FAISS vector store (created by ingest.py)
│   ├── index.faiss
│   └── index.pkl
├── chat_memory.json          # Conversation history
│
└── 🧪 Testing & Debugging
   ├── test_retrieval.py       # Debug retrieval pipeline
   └── test_full_rag.py        # Full RAG system test
```

## ⚙️ Configuration

### Environment Variables (.env)
```env
# Required: At least one API key needed
OPENAI_API_KEY=your_key_here        # Primary LLM
GOOGLE_API_KEY=your_key_here        # Fallback LLM
```

### Retrieval Parameters (retrieval.py)
```python
FAISS_SCORE_THRESHOLD = 1.0          # L2 distance threshold
COARSE_K = 10                        # Documents for initial retrieval
RERANK_TOP_K = 4                     # Final documents to use
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

### LLM Settings (chatbot.py, app.py)
```python
model="gpt-4.1"                      # Model name
temperature=0.1                      # Low temp for factual responses
```

## 🔄 How the RAG Pipeline Works

### 1. Document Ingestion (ingest.py)
```
PDFs → Load → Split into chunks → Embed → FAISS Index
        (1000 chars)           (all-MiniLM)
```

### 2. Query Processing (retrieval.py)
```
User Query
    ↓
[Stage 1] FAISS Search (k=10)
    ↓ (Filter by L2 distance < 1.0)
[Stage 2] CrossEncoder Re-ranking
    ↓ (Semantic relevance scoring)
Top 4 Documents
    ↓
Build Context String
```

### 3. Response Generation (chatbot.py)
```
System Prompt + Context + History + Query
    ↓
LLM (GPT-4.1 or Gemini)
    ↓
Response Text + Tool Calls
    ↓
Tool Execution (if needed)
    ↓
User Response + Memory Update
```

## 🧪 Testing & Debugging

### Test Retrieval Pipeline
```bash
python test_retrieval.py
```
Shows what documents are retrieved for test queries.

### Test Full RAG System
```bash
python test_full_rag.py
```
Tests retrieval + LLM response for multiple queries.

### Debug Output
Both CLI and Streamlit show:
- Number of documents retrieved
- Context size in characters
- Response generation status

## 📊 Example Queries

### Financial Analysis
- "What were the total revenues for FY 2024?"
- "List the company's key financial metrics"
- "Explain the profit margin trends"

### Data Visualization
- "Create a pie chart showing revenue by segment"
- "Show a bar chart of quarterly performance"
- "Visualize the expense breakdown"

### Document Summarization
- "Summarize the company's key achievements"
- "What are the main risks identified?"
- "Explain the geographic revenue distribution"

### Report Generation
- "Save a comprehensive financial summary as PDF"
- "Generate a report on operational performance"
- "Create a PDF with all key metrics"

## 🛠️ Troubleshooting

### Issue: "No documents found in vectordb"
**Solution**: Run `python ingest.py` to create the vector database.

### Issue: API Key Errors
**Solution**:
- Check `.env` file exists in project root
- Verify key format: `OPENAI_API_KEY=sk-proj-...`
- Ensure no extra spaces or quotes

### Issue: PDF Generation Fails
**Solution**:
- Ensure `outputs/` directory is writable
- Check disk space available
- Verify report text doesn't exceed limits

### Issue: Chart Rendering Error
**Solution**:
- Chart.js blocks are only rendered in web interface
- PDF reports show text descriptions instead
- Use Streamlit app for interactive charts

### Issue: Slow Response Times
**Solution**:
- First run loads embedding model (~90MB download)
- Subsequent runs use cached model
- Re-ranking takes ~2-3 seconds for 10 documents

## 🔐 Security & Privacy

- **Local Processing**: All embeddings computed locally
- **No Cloud Storage**: Documents never uploaded to external servers
- **API Keys Only**: Only API calls sent to LLM providers
- **Memory Persistence**: Chat history stored locally in JSON
- **No Telemetry**: No usage tracking or analytics

## 📈 Performance Notes

| Metric | Value |
|--------|-------|
| Embedding Model Size | ~90 MB |
| FAISS Index Size | ~2 MB (per 100 pages) |
| Query Response Time | 3-8 seconds |
| Token Limit (GPT-4.1) | 128,000 |
| Max PDF Size | Practical: <500 pages |

## 🚀 Next Steps & Roadmap

### Planned Enhancements
- [ ] Support for more document types (DOCX, TXT, MD, HTML)
- [ ] Advanced filtering (date range, document type)
- [ ] Custom prompt templates
- [ ] Conversation export (TXT, PDF, Markdown)
- [ ] Document metadata display
- [ ] Advanced analytics dashboard
- [ ] Multi-user support with authentication
- [ ] API endpoint for programmatic access

### Architecture Improvements
- [ ] Implement query expansion for better retrieval
- [ ] Add document clustering for organization
- [ ] Hybrid search (semantic + keyword)
- [ ] Caching layer for frequent queries
- [ ] Async processing for large documents
- [ ] Database backend (PostgreSQL) for persistence

## 📝 Notes for Developers

### Adding New Tools
1. Create tool in `tools.py` with `@tool` decorator
2. Add to `ALL_TOOLS` list
3. Add instructions in `prompts.py`
4. Update `_dispatch_tool()` in `chatbot.py` and `app.py`

### Modifying Prompts
- Edit `prompts.py` for system messages
- Change `DOCUBOT_BASE_PERSONA` for agent personality
- Adjust instructions in `build_rag_system_message()`

### Tuning Retrieval
- `COARSE_K`: Increase for broader search, decrease for speed
- `RERANK_TOP_K`: Control final context size
- `FAISS_SCORE_THRESHOLD`: Adjust relevance filter

## 📄 File Descriptions

| File | Purpose |
|------|---------|
| `ingest.py` | Load PDFs, create embeddings, build FAISS index |
| `chatbot.py` | CLI interface with conversation loop |
| `app.py` | Streamlit web UI with document upload |
| `retrieval.py` | Two-stage retrieval pipeline |
| `memory.py` | Chat history management |
| `tools.py` | Tool definitions (PDF generation) |
| `prompts.py` | System prompts and instructions |
| `utils.py` | Helper functions |
| `requirements.txt` | Python package dependencies |

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

[Your License Here]

## 👨‍💻 Author

Created as an internship project for Regan Technologies

## 📞 Support

For issues, questions, or suggestions:
- Check the Troubleshooting section
- Review test files for examples
- Check `.env` configuration
- Verify API keys are valid

## 🎯 Key Takeaways

- **DocuBot is a local-first RAG chatbot** - your data stays private
- **Two-stage retrieval** ensures accurate, relevant answers
- **Dual interfaces** support both CLI and web usage
- **Tool integration** enables PDF report generation
- **Persistent memory** maintains conversation context
- **Modular design** makes customization straightforward

---

**Happy analyzing! 📚🔍**
