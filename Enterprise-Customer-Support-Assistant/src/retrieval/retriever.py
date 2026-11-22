
from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.chroma_store import ChromaVectorStore


class Retriever:
    """
    Retrieve relevant document chunks using ChromaDB.
    """

    def __init__(self, top_k=3):
        self.top_k = top_k

        print("Initializing Retriever...")

        self.embedding_model = EmbeddingModel()
        self.vector_store = ChromaVectorStore()

        print("Retriever initialized successfully!")

    def retrieve(self, question):
        """
        Search ChromaDB using a customer question.
        """

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        results = self.vector_store.search(
            query=question,
            embedding_model=self.embedding_model,
            top_k=self.top_k
        )

        formatted_results = []

        for result in results:
            distance = result["distance"]

            # Convert cosine distance into cosine similarity.
            similarity = 1.0 - distance

            formatted_results.append({
                "content": result["content"],
                "metadata": result["metadata"],
                "similarity": similarity
            })

        return formatted_results

    def close(self):
        self.vector_store.client.close()


if __name__ == "__main__":

    retriever = Retriever(top_k=2)

    try:
        question = "How can a customer get a refund?"

        print("\nCustomer question:", question)

        results = retriever.retrieve(question)

        print("\nRetrieved documents:")

        if not results:
            print("No relevant documents found.")

        for index, result in enumerate(results, start=1):
            print(f"\nResult {index}")
            print("Content:", result["content"])
            print("Source:", result["metadata"]["source"])
            print("Page:", result["metadata"]["page"])
            print("Similarity:", round(result["similarity"], 4))

    finally:
        retriever.close()
