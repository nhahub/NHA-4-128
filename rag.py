from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "data" / "kb"
PERSIST_DIR = BASE_DIR / ".faiss"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# =========================================================
# Cache
# =========================================================

_embeddings = None
_vectorstore = None

# =========================================================
# Embeddings
# =========================================================

def get_embeddings():
    """
    Load the embedding model once.
    """

    global _embeddings

    if _embeddings is None:
        print("[RAG] Loading embedding model...")

        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True
            },
        )

        print("[RAG] Embedding model loaded.")

    return _embeddings


# =========================================================
# Vector Store
# =========================================================

def build_default_vectorstore():
    """
    Load the FAISS vector database.

    Returns:
        FAISS object if found.
        None if database doesn't exist.
    """

    global _vectorstore

    # Already loaded
    if _vectorstore is not None:
        return _vectorstore

    if not PERSIST_DIR.exists():
        print(f"[RAG] Vectorstore not found: {PERSIST_DIR}")
        return None

    try:

        print("[RAG] Loading FAISS database...")

        _vectorstore = FAISS.load_local(
            folder_path=str(PERSIST_DIR),
            embeddings=get_embeddings(),
            allow_dangerous_deserialization=True,
        )

        print("[RAG] Vectorstore loaded successfully.")

        return _vectorstore

    except Exception as e:

        print(f"[RAG] Failed to load vectorstore")

        print(e)

        return None


# =========================================================
# Retriever
# =========================================================

def get_retriever(k: int = 4):
    """
    Return a retriever from the loaded vectorstore.
    """

    vectorstore = build_default_vectorstore()

    if vectorstore is None:
        return None

    return vectorstore.as_retriever(
        search_kwargs={
            "k": k
        }
    )