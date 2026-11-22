from src.retrieval.hybrid_retriever import HybridRetriever
from src.llm.ollama_client import ask_qwen
from src.llm.answer_verifier import verify_answer, FALLBACK_MESSAGE


class RAGChain:
    """
    Document-based RAG pipeline with:
    - Hybrid retrieval
    - Optional document filtering
    - Answer generation
    - Answer verification
    - Source tracking
    """

    def __init__(self, top_k=8):
        print("Initializing RAG Chain...")

        self.retriever = HybridRetriever(top_k=top_k)

        print("RAG Chain initialized successfully!")

    def ask(self, question, selected_sources=None):

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        print("\nSearching the knowledge base...")

        # Retrieve from all documents or selected documents
        results = self.retriever.retrieve(
            question,
            selected_sources=selected_sources
        )

        if not results:
            return {
                "answer": FALLBACK_MESSAGE,
                "sources": []
            }

        context_parts = []
        sources = []
        seen_sources = set()

        for result in results:

            metadata = result.get("metadata", {})

            source_name = metadata.get("source", "Unknown document")
            page_number = metadata.get("page", "Unknown")

            context_parts.append(
                f"Source: {source_name}\n"
                f"Page: {page_number}\n"
                f"Content: {result['content']}"
            )

            source_key = (source_name, page_number)

            if source_key not in seen_sources:
                seen_sources.add(source_key)

                sources.append({
                    "source": source_name,
                    "page": page_number,
                    "hybrid_score": result.get(
                        "hybrid_score", 0.0
                    )
                })

        context = "\n\n".join(context_parts)

        system_prompt = """
You are a helpful document-based AI assistant.

Answer questions using the provided document context.

STRICT RULES:

1. Use only the supplied document context for factual answers.
2. Never invent facts, names, courses, policies, dates,
   figures, procedures, or explanations.
3. Answer the user's question directly.
4. When asked for a list, include all relevant items
   explicitly present in the provided context.
5. Distinguish mandatory subjects from elective choices
   when the document makes that distinction.
6. If the context is incomplete, clearly explain that
   the available information may be incomplete.
7. If the information is unavailable, say so clearly.
8. Never pretend to have access to information that
   was not provided in the context.
9. Do not include invented citations or page references.
10. Format answers using clear paragraphs or bullet points
    when appropriate.
11. The documents may contain academic, technical,
    business, policy, or general reference material.
12. Do not assume the documents are customer-support policies.
"""

        print("\nGenerating answer with Qwen...")

        answer = ask_qwen(
            question=f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Provide a clear answer using only the document context.
""",
            system_prompt=system_prompt
        )

        print("\n========== GENERATED ANSWER ==========")
        print(answer)
        print("=================================")

        print("\nVerifying generated answer...")

        is_supported = verify_answer(
            question=question,
            context=context,
            answer=answer
        )

        if not is_supported:

            print("Answer rejected: insufficient supporting evidence.")

            return {
                "answer": FALLBACK_MESSAGE,
                "sources": []
            }

        print("Answer successfully verified.")

        return {
            "answer": answer,
            "sources": sources
        }

    def close(self):
        self.retriever.close()


if __name__ == "__main__":

    rag = RAGChain(top_k=8)

    try:

        question = (
            "List the subjects offered in Semester 5 "
            "of the CSE AI & ML syllabus."
        )

        result = rag.ask(question)

        print("\n========== FINAL ANSWER ==========")
        print(result["answer"])

        print("\n========== VERIFIED SOURCES ==========")

        for source in result["sources"]:
            print(
                f"{source['source']} | "
                f"Page {source['page']} | "
                f"Hybrid Score: {source['hybrid_score']:.5f}"
            )

    finally:
        rag.close()