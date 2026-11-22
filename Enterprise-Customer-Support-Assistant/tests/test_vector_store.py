
import numpy as np
from langchain_core.documents import Document

from src.vectorstore.local_vector_store import LocalVectorStore


class DummyEmbeddingModel:
    """Small test model to verify vector storage and retrieval."""

    def generate_embeddings(self, texts):
        vectors = []

        for text in texts:
            if "refund" in text.lower():
                vectors.append([1.0, 0.0, 0.0])
            elif "delivery" in text.lower():
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])

        return np.array(vectors, dtype=np.float32)

    def generate_single_embedding(self, text):
        return self.generate_embeddings([text])[0]


def test_vector_store_add_and_search(tmp_path):
    database_path = tmp_path / "test_vector_store.db"

    store = LocalVectorStore(
        database_path=str(database_path)
    )

    embedding_model = DummyEmbeddingModel()

    documents = [
        Document(
            page_content="Customers can request a refund within 30 days.",
            metadata={"source": "refund.pdf", "page": 1}
        ),
        Document(
            page_content="Standard delivery takes 3 to 5 days.",
            metadata={"source": "delivery.pdf", "page": 2}
        )
    ]

    store.add_documents(documents, embedding_model)

    assert store.get_document_count() == 2

    results = store.search(
        "How can I get a refund?",
        embedding_model,
        top_k=1
    )

    assert len(results) == 1
    assert "refund" in results[0]["content"].lower()
    assert results[0]["metadata"]["source"] == "refund.pdf"

    store.close()
