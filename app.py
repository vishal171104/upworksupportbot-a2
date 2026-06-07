import streamlit as st

from rag import answer, get_llm, get_vectorstore


@st.cache_resource
def load_vectorstore():
    return get_vectorstore()


@st.cache_resource
def load_llm():
    return get_llm()


st.set_page_config(page_title="Upwork API Support Bot", page_icon="🤖")
st.title("Upwork API Support Bot")

vectorstore = load_vectorstore()
llm = load_llm()

question = st.text_input("Ask a question about the Upwork API:")

if st.button("Submit") and question.strip():
    with st.spinner("Searching documentation and generating answer..."):
        result = answer(question.strip(), vectorstore=vectorstore, llm=llm)

    st.subheader("Answer")
    st.write(result["answer"])

    with st.expander("📄 Sources"):
        for i, source in enumerate(result["sources"], start=1):
            st.markdown(f"**Source {i}**")
            st.text(source)
            if i < len(result["sources"]):
                st.divider()

    st.metric("Latency", f"{result['latency_seconds']:.2f} s")
