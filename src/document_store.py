import os
import sys
from typing import List, Mapping

import chromadb
import httpx
from chromadb import ClientAPI, GetResult
from chromadb.api.models import Collection
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from llama_index import Response
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.core.base_query_engine import BaseQueryEngine
from llama_index.llms import OpenAILike, LLM
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore

from ingest import Ingest


class DocumentStore:
    def __init__(self):
        self.client: ClientAPI = chromadb.PersistentClient(path="db")
        self.default_collection_name: str = "default"
        self.collection: Collection = self.client.get_or_create_collection(name=self.default_collection_name)
        self.chunk_size: int = int(os.environ.get('CHUNK_SIZE'))
        self.overlap_size: int = int(os.environ.get('OVERLAP_SIZE'))
        self.response_mode: str = os.environ.get('RESPONSE_MODE')
        self.context_window: int = int(os.environ.get('CONTEXT_WINDOW'))
        self.max_output: int = int(os.environ.get('MAX_OUTPUT'))
        self.temperature: float = float(os.environ.get('TEMPERATURE'))
        self.model: str = os.environ.get('MODEL')
        self.ingestor: Ingest = Ingest(self.collection, self.chunk_size, self.overlap_size)

        self.embedding_function: Embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.chroma_db: Chroma = Chroma(embedding_function=self.embedding_function,
                                        collection_name=self.default_collection_name,
                                        client=self.client)

        # LlamaIndex
        api_key: str = os.environ.get('OPENAI_API_KEY')
        api_base: str = os.environ.get('OPENAI_API_BASE')
        verify_https: bool = os.environ.get('SSL_VERIFICATION').lower() == "true"
        http_client: httpx.Client = httpx.Client(verify=verify_https)
        self.llm: LLM = OpenAILike(is_chat_model=True, model=self.model, temperature=self.temperature,
                                   max_tokens=self.context_window,
                                   api_base=api_base, api_key=api_key,
                                   http_client=http_client)
        vector_store: ChromaVectorStore = ChromaVectorStore(chroma_collection=self.collection)
        storage_context: StorageContext = StorageContext.from_defaults(vector_store=vector_store)
        service_context: ServiceContext = ServiceContext.from_defaults(embed_model=self.embedding_function,
                                                                       llm=self.llm,
                                                                       context_window=self.context_window,
                                                                       num_output=self.max_output,
                                                                       chunk_size=self.chunk_size,
                                                                       chunk_overlap=self.overlap_size)
        self.index: VectorStoreIndex = VectorStoreIndex.from_documents(
            [], storage_context=storage_context, service_context=service_context
        )

    def get_all_documents(self) -> dict[str, str]:
        all_documents: GetResult = self.collection.get(include=['metadatas'])
        all_metadatas: list[Mapping] = all_documents['metadatas']
        unique_documents: dict[str, str] = {}
        for metadata in all_metadatas:
            unique_documents[metadata['uuid']] = metadata['source']
        return unique_documents

    def delete_documents(self, uuids: List[str]):
        for uuid in uuids:
            self.collection.delete(where={'uuid': uuid})

    # Try catch fails if collection cannot be found
    def does_collection_exist(self, collection_name: str) -> bool:
        try:
            collection = self.client.get_collection(name=collection_name)
            if collection.count() > 0:
                return True
        except ValueError:
            return False

        return False

    def query_llamaindex(self, query_text: str, response_mode: str = None) -> str | None:
        if response_mode is None:
            response_mode = self.response_mode

        query_engine: BaseQueryEngine = self.index.as_query_engine(response_mode=response_mode)
        query_result: Response = query_engine.query(query_text)

        if response_mode == "no_text":
            query_response: str = query_result.get_formatted_sources(length=sys.maxsize)
        else:
            query_response: str = query_result.response

        return query_response

    def load_file_bytes(self, file_bytes: bytes, file_name: str) -> None:
        self.ingestor.load_file_bytes(file_bytes, file_name)
