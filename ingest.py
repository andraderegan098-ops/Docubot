import os
import shutil
import time
from utils import process_pdfs, get_embedding_model
from langchain_community.vectorstores import FAISS
from deep_translator import GoogleTranslator


def batch_translate_docs(docs):
    """Translate documents to English in batches (much faster)."""
    print("🌍 Translating documents to English for multilingual search...")

    texts = [doc.page_content for doc in docs]

    try:
        translated_texts = GoogleTranslator(source='auto', target='en').translate_batch(texts)
    except Exception as e:
        print(f"⚠️ Batch translation failed, using original text. Error: {e}")
        return docs

    for doc, translated in zip(docs, translated_texts):
        doc.page_content = translated

    return docs


def create_db():
    start_time = time.time()

    print("🚀 Step 1: Scanning for PDFs in 'input' folder...")
    docs = process_pdfs("input")

    if not docs:
        print("🛑 ERROR: No PDFs found. Place files in 'input/' and retry.")
        return

    print(f"📄 Found {len(docs)} text segments.")

    # --------- FAST BATCH TRANSLATION ---------
    docs = batch_translate_docs(docs)
    # -----------------------------------------

    print("🧠 Step 2: Creating embeddings and FAISS index...")

    if os.path.exists("vectordb"):
        shutil.rmtree("vectordb")

    embeddings = get_embedding_model()

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vectordb")

    end_time = time.time()
    print("✅ Success! 'vectordb' is ready.")
    print(f"⏱️ Total ingest time: {round(end_time - start_time, 2)} seconds")


if __name__ == "__main__":
    create_db()
