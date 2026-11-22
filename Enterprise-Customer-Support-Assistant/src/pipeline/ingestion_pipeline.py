from pathlib import Path

from src.preprocessing.document_loader import load_pdf_documents
from src.preprocessing.text_splitter import split_documents
from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.chroma_store import ChromaVectorStore


def run_ingestion(
    pdf_path,
    persist_directory="data/chroma_db"
):
    """
    Process one PDF and replace all existing ChromaDB knowledge
    with chunks from that PDF only.
    """

    print("\nStarting single-PDF ingestion...")

    vector_store = None

    try:
        # Step 1: Validate the uploaded PDF
        pdf_path = Path(pdf_path).resolve()

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")

        print(f"\nSelected PDF: {pdf_path.name}")

        # Step 2: Load PDF documents
        print("\nStep 1: Loading PDF...")

        all_documents = load_pdf_documents(
            str(pdf_path.parent)
        )

        # Keep pages from the selected PDF only.
        selected_documents = [
            doc for doc in all_documents
            if Path(
                str(doc.metadata.get("source", ""))
            ).name == pdf_path.name
        ]

        if not selected_documents:
            raise ValueError(
                f"No pages found for {pdf_path.name}"
            )

        print(
            f"Loaded {len(selected_documents)} page(s) "
            f"from {pdf_path.name}"
        )

        # Step 3: Split selected PDF into chunks
        print("\nStep 2: Splitting PDF into chunks...")

        chunks = split_documents(selected_documents)

        if not chunks:
            raise ValueError(
                "No valid chunks generated. "
                "Existing knowledge was not replaced."
            )

        print(f"Generated {len(chunks)} chunks.")

        # Step 4: Initialize embedding model
        print("\nStep 3: Initializing embedding model...")

        embedding_model = EmbeddingModel()

        # Step 5: Initialize ChromaDB
        print("\nStep 4: Initializing ChromaDB...")

        vector_store = ChromaVectorStore(
            persist_directory=persist_directory
        )

        # Step 6: Replace the ENTIRE knowledge base
        print("\nStep 5: Replacing existing knowledge...")

        total_processed = vector_store.replace_all_documents(
            source=pdf_path.name,
            documents=chunks,
            embedding_model=embedding_model
        )

        # Step 7: Verify database
        total_chunks = vector_store.get_document_count()
        sources = vector_store.list_sources()

        print("\nIngestion completed successfully!")
        print(f"New PDF: {pdf_path.name}")
        print(f"Processed chunks: {total_processed}")
        print(f"Chunks in ChromaDB: {total_chunks}")
        print(f"Stored PDFs: {sources}")

        return total_processed

    except Exception as error:

        print("\nDocument ingestion failed.")
        print("Error:", error)

        raise

    finally:

        if vector_store is not None:
            vector_store.close()


if __name__ == "__main__":
    print(
        "This pipeline requires the path of one PDF file. "
        "Call run_ingestion(pdf_path) from the Streamlit upload workflow."
    )
