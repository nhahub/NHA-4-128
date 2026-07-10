from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)


# Paths
BASE_DIR = Path(__file__).parent

KB_DIR = BASE_DIR / "data" / "kb"
FAISS_DIR = BASE_DIR / ".faiss"


# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    """
    Load documents from knowledge base folder
    """

    documents = []

    for file in KB_DIR.iterdir():

        if file.suffix.lower() == ".txt":
            loader = TextLoader(
                str(file),
                encoding="utf-8"
            )

        elif file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(
                str(file)
            )

        else:
            continue

        docs = loader.load()

        documents.extend(docs)

    return documents
def load_documents():
    """
    Load documents from knowledge base folder
    """

    documents = []

    for file in KB_DIR.iterdir():

        if file.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(
                str(file),
                encoding="utf-8"
            )

        elif file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(
                str(file)
            )

        else:
            continue

        docs = loader.load()
        documents.extend(docs)

    return documents


def build_vector_store():

    print("Loading documents...")

    docs = load_documents()

    if not docs:
        raise Exception(
            "No documents found inside data/kb"
        )


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


    # Save
    vector_db.save_local(
        str(FAISS_DIR)
    )


    print(
        "FAISS knowledge base created successfully!"
    )



if __name__ == "__main__":
    build_vector_store()