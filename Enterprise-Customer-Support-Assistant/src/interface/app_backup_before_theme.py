
import sys
from pathlib import Path

import streamlit as st

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.rag_chain import RAGChain


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise Customer Support Assistant",
    page_icon="💬",
    layout="centered"
)


# --------------------------------------------------
# INITIALIZE RAG CHAIN
# --------------------------------------------------

@st.cache_resource
def initialize_rag():
    return RAGChain(top_k=3)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("💬 Customer Support Assistant")

st.caption(
    "Enterprise Knowledge Assistant powered by "
    "Hybrid Retrieval, RAG and Qwen 2.5"
)

st.divider()

st.info(
    "Ask questions about company policies, refunds, "
    "delivery and account support."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("About")

    st.write(
        "This assistant retrieves information from "
        "company documents and generates answers "
        "using a local Qwen language model."
    )

    st.write("**Technology Stack:**")

    st.markdown(
        """
        - Python
        - Streamlit
        - ChromaDB
        - Ollama Embeddings
        - BM25
        - Hybrid Retrieval
        - Qwen 2.5
        """
    )

    st.divider()

    if st.button(
        "Clear Chat History",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("View Sources"):

                for source in message["sources"]:

                    st.write(
                        f"📄 {source['source']} "
                        f"| Page {source['page']}"
                    )


# --------------------------------------------------
# CUSTOMER INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask your customer support question..."
)


if question:

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                rag = initialize_rag()

                result = rag.ask(question)

                answer = result["answer"]
                sources = result["sources"]

            st.markdown(answer)

            if sources:

                with st.expander("View Sources"):

                    for source in sources:

                        st.write(
                            f"📄 {source['source']} "
                            f"| Page {source['page']}"
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as error:

            error_message = (
                "Sorry, the assistant could not process your "
                "question. Please check that Ollama is running "
                "and try again."
            )

            st.error(error_message)

            print("Application error:", error)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": []
            })


