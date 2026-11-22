
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"
REQUEST_TIMEOUT = 120


def ask_qwen(question, system_prompt=None):
    """
    Send a question to the local Qwen model through Ollama.
    Supports optional system instructions.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    messages.append({
        "role": "user",
        "content": question
    })

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        result = response.json()
        answer = result.get("message", {}).get("content")

        if not answer:
            raise ValueError("Ollama returned an empty answer.")

        return answer.strip()

    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Ollama request timed out. Please try again."
        )

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to Ollama. "
            "Make sure Ollama is running."
        )


if __name__ == "__main__":

    question = input("Enter your question: ")

    try:
        answer = ask_qwen(question)

        print("\nQwen's Answer:")
        print(answer)

    except Exception as error:
        print("\nError:", error)
