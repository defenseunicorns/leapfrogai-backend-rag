import os

import chromadb
import httpx
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from llama_index import Response
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAILike
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore

import ingest


class DocumentStore:
    def __init__(self):
        self.index_name = "default"
        self.client = chromadb.PersistentClient(path="db")
        self.collection = self.client.get_or_create_collection(name="default")
        self.ingestor = ingest.Ingest(self.index_name, self.client, self.collection)
        self.chunk_size = 1024
        self.overlap_size = 20

        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.chroma_db = Chroma(embedding_function=self.embedding_function, collection_name="default",
                                client=self.client)

        # LlamaIndex
        api_key: str = os.environ.get('OPENAI_API_KEY')
        api_base: str = os.environ.get('OPENAI_API_BASE')
        verify_https = False
        http_client = httpx.Client(verify=verify_https)
        self.llm = OpenAILike(is_chat_model=True, model="llamacpp", temperature=0.2, max_tokens=8192,
                              api_base=api_base, api_key=api_key,
                              http_client=http_client)
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        service_context = ServiceContext.from_defaults(embed_model=self.embedding_function, llm=self.llm,
                                                       context_window=8192, num_output=1024, chunk_size=self.chunk_size,
                                                       chunk_overlap=self.overlap_size)
        self.index = VectorStoreIndex.from_documents(
            [], storage_context=storage_context, service_context=service_context
        )

    # Try catch fails if collection cannot be found
    def does_collection_exist(self, collection_name):
        try:
            collection = self.client.get_collection(name=collection_name)
            if collection.count() > 0:
                return True
        except ValueError:
            return False

        return False

    def query_llamaindex(self, query_text):
        query_engine = self.index.as_query_engine(response_mode="refine")
        query_result: Response = query_engine.query(query_text)

        return query_result.response
