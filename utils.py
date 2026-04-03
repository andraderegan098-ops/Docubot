import os
import logging
import warnings
from dotenv import load_dotenv

# Core LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --------- MULTILINGUAL IMPORTS ADDED ---------
from langdetect import detect
from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile
# ---------------------------------------------

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


# --------- MULTILINGUAL FUNCTIONS ADDED ---------
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
# ------------------------------------------------


def get_embedding_model():
    """Returns a lightweight embedding model with normalized embeddings."""
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def process_pdfs(directory):
    """Reads PDFs from 'input' and splits them into chunks."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        return []

    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for file in os.listdir(directory):
        if file.endswith(".pdf"):
            file_path = os.path.join(directory, file)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load_and_split(text_splitter)

                # --------- MULTILINGUAL TRANSLATION ADDED ---------
                for doc in docs:
                    doc.page_content = translate_to_english(doc.page_content)
                # --------------------------------------------------

                all_docs.extend(docs)

            except Exception as e:
                print(f"⚠️ Error reading {file}: {e}")
    return all_docs
