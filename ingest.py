import os

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCS_PATH = os.getenv("DOCS_PATH")
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "upwork_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    if not DOCS_PATH:
        raise ValueError("DOCS_PATH environment variable is not set.")

    if "path/to/" in DOCS_PATH:
        raise ValueError(
            "DOCS_PATH still uses the .env.example placeholder. "
            "Edit .env and set DOCS_PATH to the real Upwork API documentation PDF."
        )

    if not os.path.isfile(DOCS_PATH):
        raise FileNotFoundError(
            f"Documentation file not found: {DOCS_PATH}\n"
            "Update DOCS_PATH in .env to point to your Upwork API documentation PDF."
        )

    print(f"Loading PDF from: {DOCS_PATH}")
    loader = PyPDFLoader(DOCS_PATH)
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
    except (ValueError, chromadb.errors.NotFoundError):
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
