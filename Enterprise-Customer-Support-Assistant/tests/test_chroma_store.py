
import numpy as np
from langchain_core.documents import Document

from src.vectorstore.chroma_store import ChromaVectorStore


class DummyEmbeddingModel:
    """Deterministic embeddings for testing ChromaDB."""

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


def test_chroma_add_and_search(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_test_db"),
        collection_name="test_customer_support"
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
    assert results[0]["metadata"]["page"] == 1

    store.client.close()



def test_chroma_search_empty_collection(tmp_path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "empty_chroma_db"),
        collection_name="empty_test_collection"
    )

    embedding_model = DummyEmbeddingModel()

    results = store.search(
        "How can I get a refund?",
        embedding_model
    )

    assert results == []
    assert store.get_document_count() == 0

    store.client.close()

