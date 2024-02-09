import os

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import TokenTextSplitter
from transformers import GPT2TokenizerFast

model_name = os.environ.get("EMBEDDING_MODEL_NAME")
cache_folder = "embedding-cache"
embeddings: Embeddings = SentenceTransformerEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    cache_folder=cache_folder
)

tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_model_name_or_path="gpt2", cache_dir="tokenizer-cache")
# disallowed_special is set so that technical documents that contain special tokens can be loaded
text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
    tokenizer
)
