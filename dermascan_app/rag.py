from pathlib import Path
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

KB_DIR          = Path(__file__).parent / "data" / "kb"
PERSIST_DIR     = Path(__file__).parent / ".faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 100


def load_kb(kb_dir: Path = KB_DIR) -> List[Document]:
    if not kb_dir.exists():
        raise FileNotFoundError(f"KB directory not found: {kb_dir}")
    loader = DirectoryLoader(
        str(kb_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    if not docs:
        raise RuntimeError(f"No markdown files found in {kb_dir}")
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(docs)


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_default_vectorstore() -> FAISS:
    embeddings = get_embeddings()

    if PERSIST_DIR.exists():
        try:
            vs = FAISS.load_local(
                str(PERSIST_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            print("✅ Loaded vectorstore from disk.")
            return vs
        except Exception as e:
            print(f"⚠️ Could not load from disk: {e}. Rebuilding...")

    print("🔄 Building vectorstore from KB...")
    docs   = load_kb()
    chunks = chunk_documents(docs)
    vs     = FAISS.from_documents(chunks, embeddings)
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(PERSIST_DIR))
    print(f"✅ Vectorstore saved to {PERSIST_DIR}")
    return vs


if __name__ == "__main__":
    print("Building vectorstore...")
    vs = build_default_vectorstore()
    print("\nTest query: 'risk factors for melanoma'")
    results = vs.similarity_search("risk factors for melanoma", k=3)
    for i, r in enumerate(results, 1):
        source = r.metadata.get("source", "?").split("/")[-1]
        print(f"\n[{i}] {source}")
        print(f"    {r.page_content[:150]}...")
