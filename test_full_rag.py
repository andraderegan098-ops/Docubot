"""Test full RAG pipeline with sample queries."""
import os
from dotenv import load_dotenv
from utils import get_embedding_model
from retrieval import load_vectorstore, load_reranker, retrieve_with_rerank, build_context_string
from prompts import build_rag_system_message
from tools import ALL_TOOLS
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# Test queries
test_queries = [
    "What are the main financial highlights from the annual report?",
    "What was the revenue for 2024?",
    "What are the company's key achievements?",
    "What are the risks mentioned in the report?",
]

print("=" * 80)
print("🧪 FULL RAG PIPELINE TEST")
print("=" * 80)

# Load components
print("\n📦 Loading components...")
embeddings = get_embedding_model()
vectorstore = load_vectorstore(embeddings)
reranker = load_reranker()

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)
llm_with_tools = llm.bind_tools(ALL_TOOLS)
print("✅ All components loaded\n")

# Test each query
for i, query in enumerate(test_queries, 1):
    print("=" * 80)
    print(f"\n🔍 TEST {i}: {query}\n")

    # Step 1: Retrieve documents
    print("📄 RETRIEVAL STEP:")
    relevant_docs = retrieve_with_rerank(vectorstore, query, reranker)

    if not relevant_docs:
        print("   ❌ No documents retrieved!")
        continue

    print(f"   ✅ Retrieved {len(relevant_docs)} documents\n")

    # Show retrieved documents
    for j, doc in enumerate(relevant_docs, 1):
        snippet = doc.page_content[:200].replace('\n', ' ')
        print(f"   [{j}] {snippet}...")

    # Step 2: Build context
    context = build_context_string(relevant_docs)
    print(f"\n   📊 Context size: {len(context)} characters\n")

    # Step 3: Build messages
    system_prompt = build_rag_system_message(context, include_chart_instructions=False)
    messages = [system_prompt, HumanMessage(content=query)]

    # Step 4: Get LLM response
    print("🤖 LLM RESPONSE:")
    try:
        response = llm_with_tools.invoke(messages)

        # Extract text
        if isinstance(response.content, str):
            answer = response.content
        elif isinstance(response.content, list):
            answer = ""
            for part in response.content:
                if isinstance(part, dict) and 'text' in part:
                    answer += part['text']
                elif isinstance(part, str):
                    answer += part
        else:
            answer = str(response.content)

        print(f"\n{answer}\n")

        # Check for tool calls
        if response.tool_calls:
            print(f"   🔧 Tool calls: {[tc['name'] for tc in response.tool_calls]}")

    except Exception as e:
        print(f"   ❌ Error: {e}\n")

print("\n" + "=" * 80)
print("✅ Test complete!")
print("=" * 80)
