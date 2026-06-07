import os
import subprocess
import sys

import streamlit as st

from config import CHROMA_PATH, COLLECTION_NAME, sync_streamlit_secrets

sync_streamlit_secrets()

from rag import answer, get_document_count, get_llm, get_vectorstore


def run_ingestion() -> None:
    env = os.environ.copy()
    if "DOCS_PATH" in st.secrets:
        env["DOCS_PATH"] = str(st.secrets["DOCS_PATH"])
    if "DEEPINFRA_API_KEY" in st.secrets:
        env["DEEPINFRA_API_KEY"] = str(st.secrets["DEEPINFRA_API_KEY"])

    result = subprocess.run(
        [sys.executable, "ingest.py"],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        st.error("Ingestion failed.")
        st.code(result.stderr or result.stdout)
        st.stop()


@st.cache_resource
def load_vectorstore():
    return get_vectorstore()


@st.cache_resource
def load_llm():
    return get_llm()


st.set_page_config(page_title="Upwork API Support Bot", page_icon="🤖")
st.title("Upwork API Support Bot")

if not os.path.isdir(CHROMA_PATH):
    with st.spinner("Loading knowledge base..."):
        run_ingestion()

vectorstore = load_vectorstore()
llm = load_llm()

doc_count = get_document_count()
st.caption(f"Knowledge base: {doc_count} document(s) indexed")

if doc_count == 0:
    st.warning("Knowledge base is empty, please check ingestion")

question = st.text_input("Ask a question about the Upwork API:")

if st.button("Submit") and question.strip():
    with st.spinner("Searching documentation and generating answer..."):
        result = answer(question.strip(), vectorstore=vectorstore, llm=llm)

    if not result["sources"]:
        st.warning("Knowledge base is empty, please check ingestion")

    st.subheader("Answer")
    st.write(result["answer"])

    with st.expander("📄 Sources"):
        for i, source in enumerate(result["sources"], start=1):
            st.markdown(f"**Source {i}**")
            st.text(source)
            if i < len(result["sources"]):
                st.divider()

    st.metric("Latency", f"{result['latency_seconds']:.2f} s")
