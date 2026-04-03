"""Debug script to test RAG retrieval pipeline."""
import os
from utils import get_embedding_model
from retrieval import load_vectorstore, load_reranker, retrieve_with_rerank, build_context_string

# Test query
test_query = "What are the main financial highlights from the annual report?"

print("🔍 Testing RAG Retrieval Pipeline\n")

# Load embeddings
print("1️⃣ Loading embedding model...")
embeddings = get_embedding_model()
print("✅ Embeddings loaded\n")

# Load vectorstore
print("2️⃣ Loading vector store...")
vectorstore = load_vectorstore(embeddings)
print(f"✅ Vector store loaded\n")

# Load reranker
print("3️⃣ Loading re-ranker...")
reranker = load_reranker()
print("✅ Re-ranker loaded\n")

# Test coarse retrieval
print("4️⃣ Testing coarse FAISS retrieval...")
docs_with_scores = vectorstore.similarity_search_with_score(test_query, k=10)
print(f"📄 Found {len(docs_with_scores)} documents")
for i, (doc, score) in enumerate(docs_with_scores[:5], 1):
    print(f"   [{i}] Score: {score:.4f}")
    print(f"       Content: {doc.page_content[:100]}...\n")

# Test full retrieval with re-ranking
print("5️⃣ Testing retrieval with re-ranking...")
relevant_docs = retrieve_with_rerank(vectorstore, test_query, reranker)
print(f"✅ Re-ranked to {len(relevant_docs)} top documents\n")

for i, doc in enumerate(relevant_docs, 1):
    print(f"[{i}] {doc.page_content[:150]}...\n")

# Build context
context = build_context_string(relevant_docs)
print(f"6️⃣ Final context string length: {len(context)} chars")
print(f"\n--- CONTEXT PREVIEW ---")
print(context[:500])
print("...\n")

print("✅ Retrieval pipeline test complete!")
