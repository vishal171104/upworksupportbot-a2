import time

import chromadb
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from config import CHROMA_PATH, COLLECTION_NAME, get_env
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_BASE_URL = "https://api.deepinfra.com/v1/openai"
LLM_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

SYSTEM_PROMPT  = """You are a Senior Upwork API Consultant. 
You MUST answer questions using ONLY the exact information from the CONTEXT provided below.
Do NOT use any prior knowledge or training data.
Do NOT summarize or paraphrase beyond what the context says.
If the context contains the answer, quote the relevant detail directly and be specific.
If the answer is not clearly stated in the context, respond with exactly:
'I'm sorry, but the provided documentation does not contain that information.'

CONTEXT:
{context}
"""

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore() -> Chroma:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )


def get_document_count() -> int:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        return collection.count()
    except Exception:
        return 0


def get_llm() -> ChatOpenAI:
    api_key = get_env("DEEPINFRA_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPINFRA_API_KEY is not set. Configure it in .env or Streamlit secrets."
        )

    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=api_key,
        model=LLM_MODEL,
        temperature=0,
    )


def retrieve(query: str, vectorstore: Chroma | None = None, k: int = 3) -> list[dict]:
    store = vectorstore or get_vectorstore()
    docs = store.similarity_search(query, k=k)

    return [
        {
            "text": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]


def answer(
    query: str,
    vectorstore: Chroma | None = None,
    llm: ChatOpenAI | None = None,
) -> dict:
    start = time.perf_counter()

    store = vectorstore or get_vectorstore()
    model = llm or get_llm()

    chunks = retrieve(query, vectorstore=store, k=3)
    context = "\n\n---\n\n".join(chunk["text"] for chunk in chunks)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Documentation context:\n\n{context}\n\n"
                f"Developer question: {query}"
            )
        ),
    ]

    response = model.invoke(messages)
    latency_seconds = time.perf_counter() - start

    return {
        "answer": response.content,
        "sources": [chunk["text"] for chunk in chunks],
        "latency_seconds": latency_seconds,
    }
