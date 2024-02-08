from chromadb import EmbeddingFunction, Documents, Embeddings
from langchain_core.embeddings import Embeddings as HFEmbeddings


class PassThroughEmbeddingsFunction(EmbeddingFunction[Documents]):
    def __init__(
            self,
            model_name: str = "WhereIsAI/UAE-Large-V1",
            cache_folder: str = "",
            embeddings: HFEmbeddings = None
    ):
        self.model_name = model_name
        self.cache_location = cache_folder
        self.embeddings = embeddings

    def __call__(self, input: Documents) -> Embeddings:
        return self.embeddings.embed_documents(input)
