import os

from langchain.text_splitter import TokenTextSplitter

os.environ["TIKTOKEN_CACHE_DIR"] = "tokenizer-cache"
text_splitter = TokenTextSplitter(disallowed_special=())
