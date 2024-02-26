from __future__ import annotations
import logging
import requests
import json
from attr import define, field, Factory
from typing import Optional
from griptape.tokenizers import BaseTokenizer


@define(frozen=True)
class OllamaTokenizer(BaseTokenizer):
    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_MAX_TOKENS = 8192
    TOKEN_OFFSET = 8
    EMBEDDING_ENDPOINT = "http://<dns name or ip>:11434/api/embeddings"

    model: str = field(kw_only=True, default=DEFAULT_MODEL)
    max_tokens: int = field(kw_only=True, default=Factory(lambda self: self.DEFAULT_MAX_TOKENS, takes_self=True))

    def count_tokens(self, text: str | list[dict], model: Optional[str] = None) -> int:
        """
        Handles the special case of ChatML. Implementation adopted from the official OpenAi notebook:
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        if isinstance(text, list):
            raise NotImplementedError(
                f"""token_count() is not implemented for list input. 
                See https://github.com/openai/openai-python/blob/main/chatml.md for 
                information on how messages are converted to tokens."""
            )

        # For simplicity, we assume each character in the text is a token.
        # This is not accurate and should be replaced with the actual tokenization logic.
        return len(text)

    def get_embeddings(self, text: str, model: Optional[str] = None) -> list[float]:
        model = model if model else self.model
        response = requests.post(
            self.EMBEDDING_ENDPOINT,
            data=json.dumps({"model": model, "prompt": text}),
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            raise Exception(f"Failed to get embeddings: {response.text}")

        embeddings = response.json().get("embedding", [])
        return embeddings
