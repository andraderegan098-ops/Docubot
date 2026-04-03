import os
import time
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

from retrieval import load_vectorstore, load_reranker, retrieve_with_rerank, build_context_string
from tools import ALL_TOOLS, generate_pdf_tool
from prompts import build_rag_system_message, build_no_context_system_message
from memory import load_memory, save_memory, to_langchain_messages, to_raw_dicts
from utils import get_embedding_model

def _initialize_llm():
    """Initialize LLM with OpenAI as default, Gemini as fallback."""
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY")

    if openai_key and openai_key != "your_openai_api_key_here":
        try:
            print("🔌 Using OpenAI GPT-4.1")
            return ChatOpenAI(
                model="gpt-4.1",
                temperature=0.1,
                api_key=openai_key
            )
        except Exception as e:
            print(f"⚠️ OpenAI initialization failed: {e}")
            print("🔄 Falling back to Gemini...")

    if gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            print("🔌 Using Google Gemini 2.5 Flash")
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,
                api_key=gemini_key
            )
        except Exception as e:
            raise RuntimeError(f"Both OpenAI and Gemini failed to initialize: {e}")

    raise RuntimeError(
        "❌ No valid API keys found. Set OPENAI_API_KEY or GOOGLE_API_KEY in .env file"
    )


def _extract_text(response):
    """Extract text from LLM response (handles both string and list formats)."""
    if isinstance(response.content, str):
        return response.content
    elif isinstance(response.content, list):
        text = ""
        for part in response.content:
            if isinstance(part, dict) and 'text' in part:
                text += part['text']
            elif isinstance(part, str):
                text += part
        return text
    return ""


def _dispatch_tool(tool_call, memory):
    """Execute a tool call and update memory with result."""
    if tool_call["name"] == "generate_pdf_tool":
        print("DocuBot: 📄 Generating your PDF report...")
        try:
            result = generate_pdf_tool.invoke(tool_call["args"])
            # Extract filename from result message
            success_msg = f"✅ Report successfully saved!\n\n{result}"
            print(f"\nDocuBot: {success_msg}\n")
            memory.append(AIMessage(content=success_msg))
            return True
        except FileNotFoundError as e:
            error_msg = f"❌ Error generating PDF: The output directory could not be created.\n{e}"
            print(f"\nDocuBot: {error_msg}\n")
            memory.append(AIMessage(content=error_msg))
            return False
        except ValueError as e:
            error_msg = f"❌ Error generating PDF: Invalid report content.\n{e}"
            print(f"\nDocuBot: {error_msg}\n")
            memory.append(AIMessage(content=error_msg))
            return False


def start_aria():
    """Main CLI chatbot loop."""
    try:
        embeddings = get_embedding_model()
        vectorstore = load_vectorstore(embeddings)
        reranker = load_reranker()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    except RuntimeError as e:
        print(f"❌ {e}")
        return

    try:
        llm = _initialize_llm()
    except RuntimeError as e:
        print(f"❌ {e}")
        return

    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # Initialize memory from previous conversations
    chat_history_raw = load_memory()
    memory = to_langchain_messages(chat_history_raw)

    print("\n🤖 DocuBot: Hello! I've analyzed your documents. How can I help? (Type 'exit' to stop)\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ['exit', 'quit']:
            save_memory(memory, max_turns=10)
            break

        # Two-stage retrieval: coarse search + re-ranking
        relevant_docs = retrieve_with_rerank(vectorstore, user_input, reranker)

        if not relevant_docs:
            print("⚠️ No relevant documents found")
            system_prompt = build_no_context_system_message()
        else:
            context = build_context_string(relevant_docs)
            print(f"📄 Retrieved {len(relevant_docs)} relevant documents ({len(context)} chars)")
            system_prompt = build_rag_system_message(context, include_chart_instructions=False)

        # Build message chain: [System] -> [History] -> [Current Query]
        current_query = HumanMessage(content=user_input)
        messages = [system_prompt] + memory[-6:] + [current_query]

        try:
            response = llm_with_tools.invoke(messages)
            final_text = _extract_text(response)

            # Always append user query to memory
            memory.append(current_query)

            # Handle tool calls or regular response
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    print("DocuBot: " + ("🔧 Processing your request" if tool_call["name"] == "generate_pdf_tool" else ""))
                    _dispatch_tool(tool_call, memory)
            else:
                if not final_text:
                    final_text = "I'm sorry, I couldn't process that response properly."
                print(f"\n🤖 DocuBot: {final_text}\n")
                memory.append(AIMessage(content=final_text))

        except KeyError as e:
            print(f"Error: Missing configuration key {e}. Check your environment setup.")
            time.sleep(1)
        except ValueError as e:
            print(f"Error: Invalid input or configuration {e}")
            time.sleep(1)
        except RuntimeError as e:
            print(f"Error: LLM processing failed - {e}")
            time.sleep(1)
        except Exception as e:
            print(f"Error: Unexpected error occurred - {e}")
            time.sleep(1)

if __name__ == "__main__":
    start_aria()