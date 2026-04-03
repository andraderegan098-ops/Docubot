import os
import re
import json
import tempfile
import streamlit as st
import streamlit.components.v1 as components
from audio_recorder_streamlit import audio_recorder
from ingest import create_db
from retrieval import load_vectorstore, load_reranker, retrieve_with_rerank, build_context_string
from tools import ALL_TOOLS, generate_pdf_tool
from prompts import build_rag_system_message, build_no_context_system_message
from memory import to_langchain_messages, to_raw_dicts
from utils import get_embedding_model
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langdetect import detect
from deep_translator import GoogleTranslator
from gtts import gTTS

# Configuration
CHARTJS_PATTERN = re.compile(r"```chartjs\s*([\s\S]*?)```")
MAX_UPLOADED_FILES = 10

st.set_page_config(page_title="DocuBot - Intelligent Document Analysis", layout="wide")


# ---------------- MULTILINGUAL FUNCTIONS ADDED ---------------- #
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"


def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text


def translate_from_english(text, target_lang):
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except:
        return text


def speak_text(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code)
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(fp.name)
        return fp.name
    except:
        return None
# ---------------------------------------------------------------- #


# ---------------- CHART.JS RENDER FUNCTION (FIX) ---------------- #
def render_chartjs(chart_config):
    """Render Chart.js chart inside Streamlit."""
    chart_json = json.dumps(chart_config)

    html_code = f"""
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <canvas id="myChart"></canvas>
        <script>
            const config = {chart_json};
            const ctx = document.getElementById('myChart').getContext('2d');
            new Chart(ctx, config);
        </script>
    </body>
    </html>
    """

    components.html(html_code, height=400)
# ---------------------------------------------------------------- #


def _initialize_llm():
    """Initialize LLM with OpenAI as default, Gemini as fallback."""
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY")

    if openai_key and openai_key != "your_openai_api_key_here":
        try:
            return ChatOpenAI(
                model="gpt-4.1",
                temperature=0.1,
                api_key=openai_key
            )
        except Exception as e:
            st.warning(f"OpenAI initialization failed: {e}. Using Gemini instead.")

    if gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,
                api_key=gemini_key
            )
        except Exception as e:
            st.error(f"Both OpenAI and Gemini failed: {e}")
            return None

    st.error("No valid API keys found. Set OPENAI_API_KEY or GOOGLE_API_KEY in .env")
    return None


@st.cache_resource
def load_rag_resources():
    """Load vectorstore, reranker, and LLM (cached across reruns)."""
    try:
        embeddings = get_embedding_model()
        vectorstore = load_vectorstore(embeddings)
        reranker = load_reranker()
        llm = _initialize_llm()
        if llm is None:
            return None, None, None
        llm_with_tools = llm.bind_tools(ALL_TOOLS)
        return vectorstore, reranker, llm_with_tools
    except Exception as e:
        st.error(f"Failed to load RAG resources: {e}")
        return None, None, None


@st.cache_resource
def load_whisper_model():
    """Load the Whisper speech-to-text model (cached across reruns)."""
    from faster_whisper import WhisperModel
    return WhisperModel("base", device="cpu", compute_type="int8")


def transcribe_audio(audio_bytes):
    """Transcribe audio bytes to text using Whisper."""
    model = load_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        segments, _ = model.transcribe(tmp_path)
        return "".join(segment.text for segment in segments).strip()
    finally:
        os.unlink(tmp_path)


def _extract_text(response):
    """Extract text from LLM response."""
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


def _dispatch_tool(tool_call):
    """Execute a tool call and return result string."""
    if tool_call["name"] == "generate_pdf_tool":
        try:
            result = generate_pdf_tool.invoke(tool_call["args"])
            success_msg = (
                "✅ **Report Successfully Generated!**\n\n"
                f"{result}\n\n"
                "📁 **Download**: You can find this file in the `outputs/` folder."
            )
            return success_msg
        except FileNotFoundError as e:
            return f"❌ **Error**: Could not create output directory\n\nDetails: {str(e)}"
        except ValueError as e:
            return f"❌ **Error**: Invalid report content\n\nDetails: {str(e)}"
    return "⚠️ **Warning**: Tool not recognized"


def _run_rag_turn(user_input, vectorstore, reranker, llm_with_tools):
    """Execute one RAG turn and return (response_text, tool_results)."""
    relevant_docs = retrieve_with_rerank(vectorstore, user_input, reranker)

    if not relevant_docs:
        system_prompt = build_no_context_system_message()
    else:
        context = build_context_string(relevant_docs)
        system_prompt = build_rag_system_message(context, include_chart_instructions=True)

    messages = [system_prompt] + st.session_state.get("memory", []) + [HumanMessage(content=user_input)]

    try:
        response = llm_with_tools.invoke(messages)
        final_text = _extract_text(response)

        tool_results = []
        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = _dispatch_tool(tool_call)
                tool_results.append(result)

        return final_text, tool_results
    except Exception as e:
        return f"Error: {e}", []


def _render_message(content):
    parts = CHARTJS_PATTERN.split(content)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        else:
            json_str = part.strip()
            if not json_str:
                continue

            try:
                chart_config = json.loads(json_str)
                render_chartjs(chart_config)
            except Exception as e:
                st.error(f"Chart error: {e}")


def main():
    """Main Streamlit app."""
    st.title("🤖 DocuBot - Intelligent Document Analysis")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = []

    with st.sidebar:
        st.header("📁 Document Management")

        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )

        # -------- Language Selector Added --------
        st.header("🌍 Language Settings")
        selected_lang = st.selectbox(
            "Select Response Language",
            ["auto", "en", "hi", "kn", "ta", "te", "ml", "fr", "de", "es", "zh-cn", "ja", "ar"]
        )

        if uploaded_files:
            if st.button("🔄 Ingest Documents"):
                with st.spinner("Processing documents..."):
                    input_dir = "input"
                    if not os.path.exists(input_dir):
                        os.makedirs(input_dir)

                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(input_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                    create_db()
                    st.success("Documents ingested!")

    st.header("Chat with Your Documents")

    vectorstore, reranker, llm_with_tools = load_rag_resources()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            _render_message(msg["content"])

    audio_bytes = audio_recorder(text="", icon_size="2x", pause_threshold=2.5)

    voice_input = None
    if audio_bytes:
        with st.spinner("Transcribing audio..."):
            voice_input = transcribe_audio(audio_bytes)
        if voice_input:
            st.info(f"🎤 {voice_input}")

    typed_input = st.chat_input("Ask a question about your documents...")
    user_input = voice_input or typed_input

    if user_input:
        # -------- Language Detection --------
        if selected_lang == "auto":
            user_lang = detect_language(user_input)
        else:
            user_lang = selected_lang

        user_input_en = translate_to_english(user_input)

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.memory.append(HumanMessage(content=user_input_en))

        with st.spinner("DocuBot is thinking..."):
            response_text_en, tool_results = _run_rag_turn(
                user_input_en, vectorstore, reranker, llm_with_tools
            )

        # -------- Chart-safe Translation --------
        chart_blocks = CHARTJS_PATTERN.findall(response_text_en)
        text_only = CHARTJS_PATTERN.sub("CHART_PLACEHOLDER", response_text_en)

        translated_text = translate_from_english(text_only, user_lang)

        for chart in chart_blocks:
            translated_text = translated_text.replace(
                "CHART_PLACEHOLDER",
                f"```chartjs\n{chart}\n```",
                1
            )

        response_text = translated_text

        # -------- Voice Response --------
        audio_file = speak_text(response_text, user_lang)
        if audio_file:
            st.audio(audio_file, format="audio/mp3")

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.session_state.memory.append(AIMessage(content=response_text_en))

        st.rerun()


if __name__ == "__main__":
    main()
