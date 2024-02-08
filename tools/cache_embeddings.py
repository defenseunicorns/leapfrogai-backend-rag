import os

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.embeddings import Embeddings

model_name = os.environ.get("EMBEDDING_MODEL_NAME")
cache_folder = "embedding-cache"
embeddings: Embeddings = SentenceTransformerEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    cache_folder=cache_folder
)
