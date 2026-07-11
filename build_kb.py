from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Paths
BASE_DIR = Path(__file__).parent

KB_FILE = BASE_DIR / "data" / "Kb" / "knowledge_base.md"
FAISS_DIR = BASE_DIR / ".faiss"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    """
    Load the knowledge base markdown file.
    """

    if not KB_FILE.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {KB_FILE}")

    loader = TextLoader(str(KB_FILE), encoding="utf-8")

    return loader.load()


def build_vector_store():

    print("Loading documents...")

    docs = load_documents()

    if not docs:
        raise Exception("Knowledge base file is empty")

    print(f"Loaded {len(docs)} documents")

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    # Embeddings
    print("Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Create FAISS index
    print("Building FAISS database...")

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Save vector store
    FAISS_DIR.mkdir(exist_ok=True)

    vector_db.save_local(str(FAISS_DIR))

    print("FAISS knowledge base created successfully!")


if __name__ == "__main__":
    build_vector_store()