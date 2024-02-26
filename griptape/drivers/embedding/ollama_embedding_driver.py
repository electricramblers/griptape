from __future__ import annotations
from typing import Optional
from attr import define, field, Factory
from griptape.drivers import BaseEmbeddingDriver
from griptape.tokenizers import OllamaTokenizer
import requests
import json


@define
class OllamaEmbeddingDriver(BaseEmbeddingDriver):
    """
    Attributes:
        model: Ollama embedding model name. Defaults to `nomic-embed-text`.
        base_url: API URL. Defaults to local Ollama API URL.
        tokenizer: Optionally provide custom `OllamaTokenizer`.
    """

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_BASE_URL = "http://localhost:11434/api/embeddings"

    model: str = field(default=DEFAULT_MODEL, kw_only=True, metadata={"serializable": True})
    base_url: Optional[str] = field(default=DEFAULT_BASE_URL, kw_only=True, metadata={"serializable": True})
    tokenizer: OllamaTokenizer = field(default=Factory(OllamaTokenizer, takes_self=False), kw_only=True)

    def try_embed_chunk(self, chunk: str) -> list[float]:
        response = requests.post(
            self.base_url,
            data=json.dumps({"model": self.model, "prompt": chunk}),
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            raise Exception(f"Failed to get embeddings: {response.text}")

        embeddings = response.json().get("embedding", [])
        return embeddings.to_list()
