import os

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import CHROMA_PATH, COLLECTION_NAME, get_env
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    docs_path = get_env("DOCS_PATH")
    if not docs_path:
        raise ValueError(
            "DOCS_PATH is not set. Configure it in .env or Streamlit secrets."
        )

    if "path/to/" in docs_path:
        raise ValueError(
            "DOCS_PATH still uses the .env.example placeholder. "
            "Edit .env and set DOCS_PATH to the real Upwork API documentation PDF."
        )

    if not os.path.isfile(docs_path):
        raise FileNotFoundError(
            f"Documentation file not found: {docs_path}\n"
            "Set DOCS_PATH in .env or st.secrets to your Upwork API documentation PDF."
        )

    print(f"Loading PDF from: {docs_path}")
    loader = PyPDFLoader(docs_path)
    documents = loader.load()

    full_text = "\n".join(doc.page_content for doc in documents)
    print(f"\nTotal character count: {len(full_text)}")
    print(f"\nSample text (first 500 chars):\n{full_text[:500]}\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION_NAME,
    )

    collection = client.get_collection(COLLECTION_NAME)
    print(f"Stored {collection.count()} chunks in ChromaDB at {CHROMA_PATH}")


if __name__ == "__main__":
    main()
