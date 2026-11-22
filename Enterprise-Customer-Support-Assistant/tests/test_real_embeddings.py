
from langchain_core.documents import Document

from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.local_vector_store import LocalVectorStore


def test_real_embedding_storage_and_search(tmp_path):
    database_path = tmp_path / "real_embedding_test.db"

    embedding_model = EmbeddingModel()

    vector_store = LocalVectorStore(
        database_path=str(database_path)
    )

    documents = [
        Document(
            page_content="Customers can request a refund within 30 days of purchase.",
            metadata={"source": "refund_policy.pdf", "page": 1}
        ),
        Document(
            page_content="Standard delivery usually takes 3 to 5 business days.",
            metadata={"source": "delivery_policy.pdf", "page": 2}
        ),
        Document(
            page_content="Customers can reset their account password through settings.",
            metadata={"source": "account_guide.pdf", "page": 3}
        )
    ]

    try:
        vector_store.add_documents(documents, embedding_model)

        assert vector_store.get_document_count() == 3

        results = vector_store.search(
            "How can I get my money back?",
            embedding_model,
            top_k=2
        )

        assert len(results) == 2
        assert "refund" in results[0]["content"].lower()
        assert results[0]["metadata"]["source"] == "refund_policy.pdf"

    finally:
        vector_store.close()
