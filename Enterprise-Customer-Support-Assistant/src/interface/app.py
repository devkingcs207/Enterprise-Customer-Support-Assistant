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
from src.pipeline.ingestion_pipeline import run_ingestion


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Document Knowledge Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# LEATHER OFFICE - LIGHT THEME
# --------------------------------------------------

st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #F5EEE5 !important;
    color: #38291D !important;
}

.block-container {
    padding-top: 3rem !important;
    padding-bottom: 2rem !important;
}

/* HEADER */

[data-testid="stHeader"],
[data-testid="stToolbar"],
header {
    background-color: #F5EEE5 !important;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #7B4B2A 0%,
        #684327 55%,
        #5B3823 100%
    ) !important;
    border-right: 2px solid #A77B52 !important;
}

/* SIDEBAR TEXT */

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFF8EF !important;
}

/* KEEP SIDEBAR EXPANDED */

[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarHeader"] button {
    display: none !important;
}

/* HEADINGS */

h1, h2, h3, h4 {
    color: #493421 !important;
}

/* MAIN TEXT */

.stMarkdown p,
.stMarkdown li {
    color: #493421 !important;
}

/* WELCOME BANNER */

.welcome-banner {
    background-color: #E9D7C2;
    color: #493421 !important;
    border: 1px solid #CDB397;
    border-radius: 12px;
    padding: 18px 20px;
    font-size: 16px;
    font-weight: 500;
    line-height: 1.6;
}

/* CHAT MESSAGES */

[data-testid="stChatMessage"] {
    background-color: #FFF9F1 !important;
    border: 1px solid #E2D1BD !important;
    border-radius: 16px !important;
    padding: 14px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 3px 8px rgba(91, 56, 35, 0.08) !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] li {
    color: #30271F !important;
}

/* CHAT INPUT AREA */

[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"] {
    background-color: #F5EEE5 !important;
}

[data-testid="stChatInput"] {
    background-color: #FFF9F1 !important;
    border: 1px solid #BFA184 !important;
    border-radius: 16px !important;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {
    color: #30271F !important;
    -webkit-text-fill-color: #30271F !important;
    caret-color: #80532F !important;
    background-color: #FFF9F1 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #806B58 !important;
    -webkit-text-fill-color: #806B58 !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #A47550 !important;
    box-shadow: 0 0 0 2px rgba(164, 117, 80, 0.15) !important;
}

/* BUTTONS */

.stButton button {
    background-color: #8B5E3C !important;
    color: #FFFFFF !important;
    border: 1px solid #70482D !important;
    border-radius: 10px !important;
}

.stButton button p,
.stButton button span {
    color: #FFFFFF !important;
}

.stButton button:hover {
    background-color: #A47550 !important;
    border-color: #70482D !important;
}

/* SOURCE EXPANDER */

[data-testid="stExpander"] {
    background-color: #F1E5D6 !important;
    border: 1px solid #D8C2AA !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] p,
[data-testid="stExpander"] span {
    color: #493421 !important;
}

/* DIVIDERS */

hr {
    border-color: #D9C6B3 !important;
}

/* SPINNER */

[data-testid="stSpinner"],
[data-testid="stSpinner"] p {
    color: #80532F !important;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# INITIALIZE RAG CHAIN
# --------------------------------------------------

@st.cache_resource
def initialize_rag():
    return RAGChain(top_k=8)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.title("📚 Document Knowledge Assistant")

st.caption(
    "Enterprise Knowledge Assistant powered by "
    "Hybrid Retrieval, RAG and Qwen 2.5"
)

st.divider()

st.markdown(
    """
    <div class="welcome-banner">
        Upload PDF documents and ask questions about their content.
        The assistant searches its knowledge base automatically
        and generates answers using the retrieved information.
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# SIDEBAR - KNOWLEDGE BASE UPLOAD
# --------------------------------------------------

with st.sidebar:

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.subheader("📚 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="knowledge_pdf_upload"
    )

    if uploaded_file is not None:

        st.caption(f"Selected file: {uploaded_file.name}")

        if st.button(
            "Upload & Process Document",
            use_container_width=True
        ):

            documents_dir = PROJECT_ROOT / "documents"
            documents_dir.mkdir(parents=True, exist_ok=True)

            safe_filename = Path(uploaded_file.name).name
            file_path = documents_dir / safe_filename

            try:

                # Save uploaded PDF
                with open(file_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())

                st.success("PDF uploaded successfully.")

                # Process ONLY the uploaded PDF
                with st.spinner(
                    "Processing document and generating embeddings..."
                ):

                    processed_chunks = run_ingestion(
                        pdf_path=str(file_path),
                        persist_directory="data/chroma_db"
                    )

                # Refresh the cached RAG chain
                initialize_rag.clear()

                # Clear chat history from the previous document
                st.session_state.messages = []

                st.success(
                    f"Knowledge replaced successfully! "
                    f"{processed_chunks} chunks processed "
                    f"from {safe_filename}."
                )

                st.rerun()

            except Exception as error:

                st.error(
                    "Document upload or processing failed."
                )

                print("Upload error:", error)


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📄 View Sources"):

                for source in message["sources"]:

                    st.write(
                        f"📄 {source['source']} "
                        f"| Page {source['page']}"
                    )


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your documents..."
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

                with st.expander("📄 View Sources"):

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
                "Sorry, the assistant could not process your question. "
                "Please check that Ollama is running and try again."
            )

            st.error(error_message)

            print("Application error:", error)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": []
            })
