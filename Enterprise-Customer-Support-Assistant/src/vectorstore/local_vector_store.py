
from pathlib import Path
from hashlib import sha256
import sqlite3

import numpy as np


class LocalVectorStore:
    def __init__(
        self,
        database_path="data/vector_store.db"
    ):
        """
        Initialize a local SQLite vector store.
        """

        project_root = Path(__file__).resolve().parents[2]
        db_path = project_root / database_path

        db_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.connection = sqlite3.connect(
            str(db_path)
        )

        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT,
                page INTEGER,
                chunk_index INTEGER,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL
            )
        """)

        self.connection.commit()

        print("SQLite Vector Store initialized successfully!")

    def add_documents(self, documents, embedding_model):
        """
        Generate embeddings and store document chunks.
        """

        if not documents:
            print("No documents available to store.")
            return

        texts = [
            doc.page_content
            for doc in documents
        ]

        embeddings = np.asarray(
            embedding_model.generate_embeddings(texts),
            dtype=np.float32
        )

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings does not match documents."
            )

        records = []

        for index, doc in enumerate(documents):

            source = str(
                doc.metadata.get("source", "unknown")
            )

            page = int(
                doc.metadata.get("page", 0)
            )

            unique_text = (
                f"{source}_{page}_{index}_{doc.page_content}"
            )

            document_id = sha256(
                unique_text.encode("utf-8")
            ).hexdigest()

            vector = embeddings[index].tobytes()

            records.append((
                document_id,
                doc.page_content,
                source,
                page,
                index,
                vector,
                embeddings.shape[1]
            ))

        with self.connection:
            self.connection.executemany("""
                INSERT OR REPLACE INTO document_chunks
                (
                    id,
                    content,
                    source,
                    page,
                    chunk_index,
                    embedding,
                    embedding_dim
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, records)

        print(
            f"Successfully stored {len(records)} document chunks!"
        )

    def search(self, query, embedding_model, top_k=3):
        """
        Find the most semantically similar document chunks.
        """

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = np.asarray(
            embedding_model.generate_single_embedding(query),
            dtype=np.float32
        )

        query_norm = np.linalg.norm(query_embedding)

        if query_norm == 0:
            raise ValueError("Query embedding has zero magnitude.")

        rows = self.connection.execute("""
            SELECT
                content,
                source,
                page,
                chunk_index,
                embedding,
                embedding_dim
            FROM document_chunks
        """).fetchall()

        if not rows:
            return []

        matches = []

        for row in rows:

            content, source, page, chunk_index, vector_blob, dimension = row

            document_embedding = np.frombuffer(
                vector_blob,
                dtype=np.float32
            )

            if dimension != len(query_embedding):
                raise ValueError(
                    "Query and document embedding dimensions do not match."
                )

            document_norm = np.linalg.norm(
                document_embedding
            )

            if document_norm == 0:
                continue

            similarity = float(
                np.dot(query_embedding, document_embedding)
                / (query_norm * document_norm)
            )

            matches.append({
                "content": content,
                "metadata": {
                    "source": source,
                    "page": page,
                    "chunk_index": chunk_index
                },
                "similarity": similarity
            })

        matches.sort(
            key=lambda item: item["similarity"],
            reverse=True
        )

        return matches[:top_k]

    def get_document_count(self):
        """
        Return the number of stored document chunks.
        """

        result = self.connection.execute(
            "SELECT COUNT(*) FROM document_chunks"
        ).fetchone()

        return result[0]

    def close(self):
        """
        Close the SQLite database connection.
        """

        if self.connection:
            self.connection.close()


if __name__ == "__main__":

    vector_store = LocalVectorStore()

    print(
        "Stored document chunks:",
        vector_store.get_document_count()
    )

    vector_store.close()
