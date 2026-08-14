import time

from genai_assignment.p1.models import RetrievedChunk


SYSTEM_PROMPT = """You are a careful RAG assistant.
Answer only from the provided context. Cite sources using bracketed source ids,
for example [acme_security]. If the context is insufficient, say you do not know.
Keep the answer concise and do not invent facts."""


class GeminiGenerator:
    def __init__(self, api_key: str, model: str) -> None:
        from google import genai

        self.model = model
        self.client = genai.Client(api_key=api_key)

    def answer(self, question: str, contexts: list[RetrievedChunk]) -> tuple[str, dict[str, int | str | None]]:
        prompt = _build_prompt(question, contexts)
        response = self._generate_with_retry(prompt)
        usage = getattr(response, "usage_metadata", None)
        return response.text.strip(), {
            "model": self.model,
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "completion_tokens": getattr(usage, "candidates_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
        }

    def _generate_with_retry(self, prompt: str):
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                return self.client.models.generate_content(model=self.model, contents=prompt)
            except Exception as exc:
                last_error = exc
                text = str(exc).lower()
                retryable = ["429", "503", "quota", "resource_exhausted", "unavailable", "high demand"]
                if not any(marker in text for marker in retryable):
                    raise
                time.sleep(min(30, 5 + attempt * 5))
        raise RuntimeError("Gemini generation failed after quota retries.") from last_error


def fallback_no_context_answer() -> str:
    return "I don't know from the provided documents."


def _build_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    context_text = "\n\n".join(
        f"[{item.source_id}] {item.text}"
        for item in contexts
    )
    return f"""{SYSTEM_PROMPT}

Context:
{context_text}

Question: {question}
Answer:"""
