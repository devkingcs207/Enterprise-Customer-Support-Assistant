
from pathlib import Path
from hashlib import sha256

import chromadb


class ChromaVectorStore:

    def __init__(
        self,
        persist_directory="data/chroma_db",
        collection_name="customer_support_knowledge"
    ):
        project_root = Path(__file__).resolve().parents[2]
        db_path = project_root / persist_directory

        db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(db_path)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print("ChromaDB initialized successfully!")

    @staticmethod
    def _generate_document_id(source, page, content):
        unique_text = f"{source}_{page}_{content}"

        return sha256(
            unique_text.encode("utf-8")
        ).hexdigest()

    # --------------------------------------------------
    # ADD DOCUMENTS
    # --------------------------------------------------

    def add_documents(self, documents, embedding_model):

        if not documents:
            print("No documents available to store.")
            return

        texts = [doc.page_content for doc in documents]

        embeddings = embedding_model.generate_embeddings(texts)

        ids = []
        metadatas = []

        for doc in documents:

            source = str(
                doc.metadata.get("source", "unknown")
            )

            page = int(
                doc.metadata.get("page", 0)
            )

            document_id = self._generate_document_id(
                source,
                page,
                doc.page_content
            )

            ids.append(document_id)

            metadatas.append({
                "source": source,
                "page": page
            })

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        print(f"Successfully stored {len(texts)} chunks!")

    # --------------------------------------------------
    # CLEAR ALL STORED DOCUMENTS
    # --------------------------------------------------

    def clear_all_documents(self):

        existing = self.collection.get(
            include=["metadatas"]
        )

        old_ids = existing.get("ids", [])

        if old_ids:
            self.collection.delete(ids=old_ids)

        print(
            f"Cleared {len(old_ids)} old chunks from ChromaDB."
        )

        return len(old_ids)

    # --------------------------------------------------
    # REPLACE ALL KNOWLEDGE WITH ONE NEW PDF
    # --------------------------------------------------

    def replace_all_documents(
        self,
        source,
        documents,
        embedding_model
    ):

        if not source:
            raise ValueError(
                "Source filename cannot be empty."
            )

        valid_documents = [
            doc for doc in documents
            if doc.page_content.strip()
        ]

        if not valid_documents:
            raise ValueError(
                f"No valid chunks found for {source}. "
                "Existing knowledge was preserved."
            )

        texts = [
            doc.page_content
            for doc in valid_documents
        ]

        # Generate embeddings BEFORE clearing old knowledge.
        # If embedding generation fails, old knowledge remains.
        embeddings = embedding_model.generate_embeddings(texts)

        ids = []
        metadatas = []

        for doc in valid_documents:

            page = int(
                doc.metadata.get("page", 0)
            )

            ids.append(
                self._generate_document_id(
                    source,
                    page,
                    doc.page_content
                )
            )

            metadatas.append({
                "source": source,
                "page": page
            })

        # Step 1: Remove ALL previous PDF chunks.
        self.clear_all_documents()

        # Step 2: Store only the new PDF.
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        print(
            f"Knowledge replacement complete! "
            f"Stored {len(ids)} chunks from {source}."
        )

        return len(ids)

    # --------------------------------------------------
    # REPLACE DOCUMENT CHUNKS (SINGLE SOURCE)
    # --------------------------------------------------

    def replace_document_chunks(
        self,
        source,
        documents,
        embedding_model
    ):

        if not source:
            raise ValueError(
                "Source filename cannot be empty."
            )

        valid_documents = [
            doc for doc in documents
            if doc.page_content.strip()
        ]

        texts = [
            doc.page_content
            for doc in valid_documents
        ]

        if not texts:
            raise ValueError(
                f"No valid chunks found for {source}. "
                "Existing records were preserved."
            )

        embeddings = embedding_model.generate_embeddings(texts)

        ids = []
        metadatas = []

        for doc in valid_documents:

            page = int(
                doc.metadata.get("page", 0)
            )

            ids.append(
                self._generate_document_id(
                    source,
                    page,
                    doc.page_content
                )
            )

            metadatas.append({
                "source": source,
                "page": page
            })

        existing = self.collection.get(
            where={"source": source},
            include=["metadatas"]
        )

        old_ids = existing["ids"]

        if old_ids:
            self.collection.delete(ids=old_ids)

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        print(
            f"Successfully replaced {len(ids)} chunks "
            f"for {source}!"
        )

    # --------------------------------------------------
    # LIST STORED PDF DOCUMENTS
    # --------------------------------------------------

    def list_sources(self):

        stored = self.collection.get(
            include=["metadatas"]
        )

        sources = set()

        for metadata in stored.get("metadatas", []):

            if metadata and metadata.get("source"):
                sources.add(metadata["source"])

        return sorted(sources)

    # --------------------------------------------------
    # SEMANTIC SEARCH
    # --------------------------------------------------

    def search(
        self,
        query,
        embedding_model,
        top_k=3,
        where=None
    ):

        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        document_count = self.collection.count()

        if document_count == 0:
            return []

        query_embedding = (
            embedding_model.generate_single_embedding(query)
            )

        query_arguments = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": min(top_k, document_count),
            "include": [
                "documents",
                "metadatas",
                "distances"
            ]
        }

        if where:
            query_arguments["where"] = where

        results = self.collection.query(
            **query_arguments
        )

        matches = []

        result_documents = results.get(
            "documents", [[]]
        )[0]

        result_ids = results.get(
            "ids", [[]]
        )[0]

        result_metadata = results.get(
            "metadatas", [[]]
        )[0]

        result_distances = results.get(
            "distances", [[]]
        )[0]

        for index, text in enumerate(result_documents):

            matches.append({
                "id": result_ids[index],
                "content": text,
                "metadata": result_metadata[index],
                "distance": result_distances[index]
            })

        return matches

    # --------------------------------------------------
    # DATABASE INFORMATION
    # --------------------------------------------------

    def get_document_count(self):
        return self.collection.count()

    def close(self):
        self.client.close()


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    vector_store = ChromaVectorStore()

    try:

        print(
            "Stored document chunks:",
            vector_store.get_document_count()
        )

        print("\nStored PDF documents:")

        for source in vector_store.list_sources():
            print("-", source)

    finally:
        vector_store.close()
