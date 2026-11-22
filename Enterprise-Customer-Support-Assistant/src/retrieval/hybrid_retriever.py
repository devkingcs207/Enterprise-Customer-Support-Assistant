
from rank_bm25 import BM25Okapi

from src.embeddings.embedding_model import EmbeddingModel
from src.vectorstore.chroma_store import ChromaVectorStore


class HybridRetriever:
    """
    Hybrid retrieval using Dense Semantic Search,
    BM25 Keyword Search, and Reciprocal Rank Fusion.

    Supports retrieval from:
    - All stored PDF documents
    - A selected list of PDF documents
    """

    def __init__(
        self,
        top_k=8,
        semantic_weight=0.6,
        keyword_weight=0.4
    ):

        print("Initializing Hybrid Retriever...")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if semantic_weight < 0 or keyword_weight < 0:
            raise ValueError("Weights cannot be negative.")

        if semantic_weight + keyword_weight == 0:
            raise ValueError("At least one weight must be positive.")

        self.top_k = top_k
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

        self.embedding_model = EmbeddingModel()
        self.vector_store = ChromaVectorStore()

        self.bm25 = None
        self.documents = []

        self._load_bm25_index()

        print("Hybrid Retriever initialized successfully!")

    # --------------------------------------------------
    # LOAD BM25 INDEX
    # --------------------------------------------------

    def _load_bm25_index(self):

        total = self.vector_store.get_document_count()

        if total == 0:

            self.bm25 = None
            self.documents = []

            print("No documents available for BM25 indexing.")
            return

        stored = self.vector_store.collection.get(
            include=["documents", "metadatas"]
            )

        self.documents = []

        for document_id, content, metadata in zip(
            stored["ids"],
            stored["documents"],
            stored["metadatas"]
        ):

            self.documents.append({
                "id": document_id,
                "content": content,
                "metadata": metadata
            })

        tokenized_documents = [
            self._tokenize(document["content"])
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

        print(
            f"BM25 index created with {len(self.documents)} chunks."
        )

    # --------------------------------------------------
    # REFRESH INDEX
    # --------------------------------------------------

    def refresh_index(self):

        print("Refreshing BM25 index...")

        self._load_bm25_index()

        print("BM25 index refreshed successfully!")

    # --------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------

    @staticmethod
    def _tokenize(text):

        return text.lower().split()

    # --------------------------------------------------
    # HYBRID RETRIEVAL
    # --------------------------------------------------

    def retrieve(self, question, selected_sources=None):
        """
        Retrieve relevant chunks using hybrid search.

        selected_sources:
            None  -> search all documents
            list  -> search only selected PDF filenames
        """

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if self.bm25 is None:
            return []

        # Explicitly empty selection means no documents.
        if selected_sources is not None and not selected_sources:
            print("No documents selected for retrieval.")
            return []

        selected_source_set = (
            set(selected_sources)
            if selected_sources is not None
            else None
        )

        # Filter BM25 documents by selected source.
        eligible_documents = [
            document
            for document in self.documents
            if (
                selected_source_set is None
                or document["metadata"].get("source")
                in selected_source_set
            )
        ]

        if not eligible_documents:
            print("No matching documents found.")
            return []

        eligible_ids = {
            document["id"]
            for document in eligible_documents
        }

        # Candidate count based on eligible documents.
        candidate_k = min(
            self.top_k * 2,
            len(eligible_documents)
        )

        # --------------------------------------------------
        # DENSE SEMANTIC SEARCH
        # --------------------------------------------------

        where_filter = None

        if selected_source_set is not None:

            where_filter = {
                "source": {
                    "$in": list(selected_source_set)
                }
            }

        semantic_results = self.vector_store.search(
            query=question,
            embedding_model=self.embedding_model,
            top_k=candidate_k,
            where=where_filter
        )

        # Keep only eligible documents.
        semantic_results = [
            result
            for result in semantic_results
            if result["id"] in eligible_ids
        ]

        # --------------------------------------------------
        # BM25 KEYWORD SEARCH
        # --------------------------------------------------

        tokenized_question = self._tokenize(question)

        keyword_scores = self.bm25.get_scores(
            tokenized_question
        )

        eligible_indices = [
            index
            for index, document in enumerate(self.documents)
            if document["id"] in eligible_ids
        ]

        keyword_ranking = sorted(
            eligible_indices,
            key=lambda index: keyword_scores[index],
            reverse=True
        )[:candidate_k]

        # --------------------------------------------------
        # SEMANTIC RANKS
        # --------------------------------------------------

        semantic_ranks = {}

        for rank, result in enumerate(
            semantic_results,
            start=1
        ):

            semantic_ranks[result["id"]] = rank

        # --------------------------------------------------
        # BM25 RANKS
        # --------------------------------------------------

        keyword_ranks = {}

        for rank, index in enumerate(
            keyword_ranking,
            start=1
        ):

            document_id = self.documents[index]["id"]

            keyword_ranks[document_id] = rank

        # --------------------------------------------------
        # RECIPROCAL RANK FUSION
        # --------------------------------------------------

        fused_scores = {}

        for document in eligible_documents:

            document_id = document["id"]

            score = 0.0

            if document_id in semantic_ranks:

                score += (
                    self.semantic_weight
                    / (60 + semantic_ranks[document_id])
                )

            if document_id in keyword_ranks:

                score += (
                    self.keyword_weight
                    / (60 + keyword_ranks[document_id])
                )

            if score > 0:

                fused_scores[document_id] = score

        # --------------------------------------------------
        # SORT RESULTS
        # --------------------------------------------------

        ranked_results = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )[:self.top_k]

        document_lookup = {
            document["id"]: document
            for document in eligible_documents
        }

        results = []

        for document_id, score in ranked_results:

            document = document_lookup[document_id]

            results.append({
                "id": document_id,
                "content": document["content"],
                "metadata": document["metadata"],
                "hybrid_score": score
            })

        return results

    # --------------------------------------------------
    # CLOSE
    # --------------------------------------------------

    def close(self):

        self.vector_store.close()


# --------------------------------------------------
# STANDALONE TEST
# --------------------------------------------------

if __name__ == "__main__":

    retriever = HybridRetriever(top_k=8)

    try:

        print("\nAvailable PDF documents:")

        sources = retriever.vector_store.list_sources()

        for index, source in enumerate(sources, start=1):
            print(f"{index}. {source}")

        print("\nInteractive Hybrid Retrieval Test")
        print("Type your question below.")
        print("Type 'exit' to stop.\n")

        while True:

            question = input("Question: ").strip()

            if question.lower() in ("exit", "quit"):
                print("Exiting retrieval test.")
                break

            if not question:
                print("Please enter a question.\n")
                continue

            results = retriever.retrieve(question)

            print("\nHybrid retrieval results:")

            if not results:

                print("No relevant documents found.")
                continue

            for index, result in enumerate(
                results,
                start=1
            ):

                print(f"\nResult {index}")
                print("Content:", result["content"])
                print("Source:", result["metadata"]["source"])
                print("Page:", result["metadata"]["page"])
                print(
                    "Hybrid score:",
                    round(result["hybrid_score"], 5)
                )

            print("\n" + "-" * 60 + "\n")

    finally:

        retriever.close()
