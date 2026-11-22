
from langchain_core.documents import Document

from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.chroma_store import ChromaVectorStore


def test_real_embeddings_with_chroma(tmp_path):

    # Initialize the real embedding model
    embedding_model = EmbeddingModel()

    # Create a temporary ChromaDB database
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "real_embedding_db"),
        collection_name="real_embedding_test"
    )

    # Prepare sample support documents
    documents = [
        Document(
            page_content="Customers can request a refund within 30 days.",
            metadata={"source": "refund.pdf", "page": 1}
        ),
        Document(
            page_content="Standard delivery takes 3 to 5 business days.",
            metadata={"source": "delivery.pdf", "page": 2}
        ),
        Document(
            page_content="Customers can reset their account password.",
            metadata={"source": "account.pdf", "page": 3}
        )
    ]

    # Store documents using real Ollama embeddings
    store.add_documents(documents, embedding_model)

    # Verify document count
    assert store.get_document_count() == 3

    # Search using a natural-language question
    results = store.search(
        "How can I get my money back?",
        embedding_model,
        top_k=1
    )

    # Verify semantic retrieval
    assert len(results) == 1
    assert "refund" in results[0]["content"].lower()
    assert results[0]["metadata"]["source"] == "refund.pdf"

    store.client.close()

