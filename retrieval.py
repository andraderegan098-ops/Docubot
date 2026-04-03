"""Two-stage retrieval pipeline: FAISS coarse search + CrossEncoder re-ranking."""
import os
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

# Configuration constants
VECTORDB_PATH = "vectordb"
FAISS_SCORE_THRESHOLD = 1.0
COARSE_K = 10
RERANK_TOP_K = 4
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_vectorstore(embeddings):
    """Load the FAISS vector store from disk.

    Args:
        embeddings: HuggingFaceEmbeddings instance

    Returns:
        FAISS vector store

    Raises:
        FileNotFoundError: If vectordb directory doesn't exist
    """
    if not os.path.exists(VECTORDB_PATH):
        raise FileNotFoundError(
            f"Vector database not found at '{VECTORDB_PATH}'. "
            "Run ingest.py to create it first."
        )
    return FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)


def load_reranker():
    """Load the CrossEncoder model for re-ranking.

    Returns:
        CrossEncoder instance

    Raises:
        RuntimeError: If model loading fails
    """
    try:
        return CrossEncoder(RERANKER_MODEL)
    except Exception as e:
        raise RuntimeError(f"Failed to load re-ranker model: {e}")


def retrieve_with_rerank(vectorstore, query, reranker):
    """Two-stage retrieval: coarse search followed by re-ranking.

    Args:
        vectorstore: FAISS vector store
        query: User query string
        reranker: CrossEncoder instance

    Returns:
        List of top re-ranked Document objects
    """
    # Stage 1: Coarse retrieval using FAISS similarity search (L2 distance, lower is better)
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=COARSE_K)
    relevant_docs = [doc for doc, _ in docs_with_scores]

    if not relevant_docs:
        return []

    # Stage 2: Re-rank using CrossEncoder (higher score = more relevant)
    pairs = [(query, doc.page_content) for doc in relevant_docs]
    scores = reranker.predict(pairs)

    # Sort by rerank score (higher is better)
    reranked = sorted(zip(relevant_docs, scores), key=lambda x: x[1], reverse=True)

    # Return top K documents
    return [doc for doc, _ in reranked[:RERANK_TOP_K]]


def build_context_string(docs):
    """Join documents into a single context string.

    Args:
        docs: List of Document objects

    Returns:
        Context string with documents separated by '---'
    """
    return "\n---\n".join([doc.page_content for doc in docs])
