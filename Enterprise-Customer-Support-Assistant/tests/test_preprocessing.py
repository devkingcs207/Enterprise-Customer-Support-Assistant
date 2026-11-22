
from langchain_core.documents import Document

from src.preprocessing.text_splitter import split_documents


def test_split_documents():
    sample_document = Document(
        page_content=(
            "Customer support helps customers solve problems. "
            "Our support team handles account issues and billing questions. "
            "Customers can contact support for technical assistance."
        ),
        metadata={
            "source": "sample.pdf",
            "page": 1
        }
    )

    chunks = split_documents(
        [sample_document],
        chunk_size=80,
        chunk_overlap=20
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.page_content.strip() != ""
        assert chunk.metadata["source"] == "sample.pdf"
        assert chunk.metadata["page"] == 1
