
import requests
import numpy as np


OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "nomic-embed-text"


class EmbeddingModel:

    def __init__(self, model_name=MODEL_NAME, batch_size=16):
        self.model_name = model_name
        self.batch_size = batch_size

        print(f"Using Ollama embedding model: {model_name}")
        print(f"Embedding batch size: {batch_size}")

    def generate_embeddings(self, texts):
        """Generate embeddings in smaller batches using Ollama."""

        if not texts:
            return np.array([], dtype=np.float32)

        all_embeddings = []

        # Process texts in batches to avoid oversized requests
        for start in range(0, len(texts), self.batch_size):

            batch = texts[start:start + self.batch_size]
            batch_number = (start // self.batch_size) + 1

            print(
                f"Embedding batch {batch_number}: "
                f"{len(batch)} text(s)"
            )

            payload = {
                "model": self.model_name,
                "input": batch
            }

            try:
                response = requests.post(
                    OLLAMA_EMBED_URL,
                    json=payload,
                    timeout=180
                )

                response.raise_for_status()

                batch_embeddings = response.json()["embeddings"]

                if len(batch_embeddings) != len(batch):
                    raise ValueError(
                        f"Expected {len(batch)} embeddings, "
                        f"received {len(batch_embeddings)}."
                    )

                all_embeddings.extend(batch_embeddings)

            except Exception as error:
                raise RuntimeError(
                    f"Embedding batch {batch_number} failed "
                    f"(texts {start + 1} to {start + len(batch)}): "
                    f"{error}"
                ) from error

        return np.array(all_embeddings, dtype=np.float32)

    def generate_single_embedding(self, text):
        """Generate an embedding for one text."""

        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        embeddings = self.generate_embeddings([text])

        return embeddings[0]


if __name__ == "__main__":

    embedding_model = EmbeddingModel()

    sample_texts = [
        "Customers can request a refund within 30 days.",
        "Standard delivery takes 3 to 5 days."
    ]

    vectors = embedding_model.generate_embeddings(sample_texts)

    print("\nEmbedding test successful!")
    print("Number of embeddings:", len(vectors))
    print("Vector dimensions:", vectors.shape)
