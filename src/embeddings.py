from typing import List, Any

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel


class PassThroughEmbeddings(BaseModel, Embeddings):

    embed_fn: Any = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self.embed_fn(texts)
        return result

    def embed_query(self, text: str) -> List[float]:
        result = self.embed_fn([text])[0]
        return result
