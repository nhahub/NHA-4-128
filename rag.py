import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

KB_DIR = Path(__file__).parent / "data" / "kb"
PERSIST_DIR = Path(__file__).parent / ".faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """Initializes and returns HuggingFace embeddings."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_default_vectorstore():
    """Loads the vector database from disk or alerts if missing."""
    embeddings = get_embeddings()
    if PERSIST_DIR.exists():
        try:
            return FAISS.load_local(str(PERSIST_DIR), embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading vectorstore: {e}")
            return None
    print("Vectorstore database not found locally.")
    return None
