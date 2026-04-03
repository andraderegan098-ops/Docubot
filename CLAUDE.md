# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DocuBot is a RAG (Retrieval-Augmented Generation) chatbot for intelligent document analysis. It ingests PDFs, builds a FAISS vector store, and answers questions using a two-stage retrieval pipeline (coarse FAISS search → CrossEncoder reranking) with GPT-4.1 as the primary LLM and Gemini 2.5 Flash as fallback.

## Commands

```bash
# Run web interface (Streamlit)
streamlit run app.py

# Run CLI interface
python chatbot.py

# Ingest documents (builds FAISS vectordb from PDFs in input/)
python ingest.py

# Test retrieval pipeline only
python test_retrieval.py

# Test full RAG pipeline end-to-end
python test_full_rag.py

# Install dependencies
pip install -r requirements.txt
```

Python version: 3.11.7

## Architecture

```
User → app.py (Streamlit web) or chatbot.py (CLI)
         ↓
       memory.py        → loads/saves chat history (chat_memory.json, last 10 turns)
         ↓
       retrieval.py     → two-stage RAG: FAISS coarse (k=10) → CrossEncoder rerank (top 4)
         ↓
       prompts.py       → builds system message with retrieved context
         ↓
       LLM (GPT-4.1, fallback: Gemini 2.5 Flash, temperature=0.1)
         ↓
       tools.py         → PDF report generation via ReportLab (outputs/)
```

## Key Configuration

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (CPU, local)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Chunking:** 1000 chars, 150 overlap (in `utils.py`)
- **FAISS threshold:** L2 distance ≤ 1.0 (in `retrieval.py`)
- **Environment:** `.env` file with `OPENAI_API_KEY` and `GOOGLE_API_KEY`

## Data Directories

- `input/` — source PDFs for ingestion
- `vectordb/` — FAISS index files (`index.faiss`, `index.pkl`)
- `outputs/` — generated PDF reports

## Key Design Decisions

- The web UI (app.py) supports Chart.js visualizations embedded in LLM responses via a `chartjs` code fence pattern, rendered as HTML components in Streamlit.
- PDF generation strips Chart.js blocks since they can't render in static PDFs.
- LLM tool binding uses LangChain's `bind_tools` pattern — the LLM decides when to call `generate_pdf_tool`.
- All embeddings are computed locally; only LLM inference calls external APIs.
