import os

from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "upwork_docs"


def sync_streamlit_secrets() -> None:
    """Copy Streamlit Cloud secrets into os.environ for subprocess and rag.py."""
    try:
        import streamlit as st

        for key in ("DEEPINFRA_API_KEY", "DOCS_PATH"):
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass


def get_env(key: str) -> str | None:
    return os.getenv(key)
