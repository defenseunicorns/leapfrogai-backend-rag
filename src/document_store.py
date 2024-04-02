import os
import sys
from typing import List, Mapping

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import httpx
from chromadb import ClientAPI, GetResult, Settings
from chromadb.api.models import Collection
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from llama_index import Response
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.core.base_query_engine import BaseQueryEngine
from llama_index.core.base_retriever import BaseRetriever
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.llms import OpenAILike, LLM
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore
from pydantic import BaseModel

from embeddings import PassThroughEmbeddings
from ingest import Ingest


class UniqueDocument(BaseModel):
    uuid: str
    source: str


class DocumentStore:
    def __init__(self, default_collection_name="default"):
        self.default_collection_name = default_collection_name
        self.client: ClientAPI = chromadb.PersistentClient(path="db", settings=Settings(anonymized_telemetry=False))
        self.embeddings_model_name = os.environ.get("EMBEDDING_MODEL_NAME") or "instructor-xl"

        self.embeddings_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            api_base=os.environ.get("OPENAI_API_BASE"),
            api_type="openai",
            model_name=os.environ.get("EMBEDDING_MODEL_NAME"),
        )

        self.embeddings: Embeddings = PassThroughEmbeddings(embed_fn=self.embeddings_function)

        self.collection: Collection = self.client.get_or_create_collection(name=default_collection_name,
                                                                           embedding_function=self.embeddings_function)
        self.chunk_size: int = int(os.environ.get('CHUNK_SIZE'))
        self.overlap_size: int = int(os.environ.get('OVERLAP_SIZE'))
        self.response_mode: str = os.environ.get('RESPONSE_MODE')
        self.context_window: int = int(os.environ.get('CONTEXT_WINDOW'))
        self.max_output: int = int(os.environ.get('MAX_OUTPUT'))
        self.temperature: float = float(os.environ.get('TEMPERATURE'))
        self.model: str = os.environ.get('MODEL')
        self.top_k: int = int(os.environ.get("TOP_K"))
        self.ingestor: Ingest = Ingest(self.collection, self.chunk_size, self.overlap_size)
        self.chroma_db: Chroma = Chroma(embedding_function=self.embeddings,
                                        collection_name=default_collection_name,
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
        self.service_context: ServiceContext = ServiceContext.from_defaults(embed_model=self.embeddings,
                                                                            llm=self.llm,
                                                                            context_window=self.context_window,
                                                                            num_output=self.max_output,
                                                                            chunk_size=self.chunk_size,
                                                                            chunk_overlap=self.overlap_size)
        self.index_dictionary: dict = {}
        self.index: VectorStoreIndex = self.construct_index_for_collection(self.default_collection_name)

    def construct_index_for_collection(self, collection_name: str) -> VectorStoreIndex:
        collection_entry: Collection = self.index_dictionary.get(collection_name)

        if collection_entry is not None:
            return collection_entry
        else:
            collection = self.client.get_or_create_collection(name=collection_name,
                                                              embedding_function=self.embeddings_function)

            vector_store: ChromaVectorStore = ChromaVectorStore(chroma_collection=collection)
            storage_context: StorageContext = StorageContext.from_defaults(vector_store=vector_store)

            self.index_dictionary[collection_name] = VectorStoreIndex.from_documents(
                [], storage_context=storage_context, service_context=self.service_context
            )

            return self.index_dictionary[collection_name]

    def get_or_create_collection(self, collection_name: str):
        return self.client.get_or_create_collection(name=collection_name,
                                                    embedding_function=self.embeddings_function)

    def get_all_documents(self, collection_name: str = None) -> list[UniqueDocument]:

        if collection_name is None:
            target_collection = self.collection
        else:
            target_collection = self.get_or_create_collection(collection_name)

        if self.collection.count() > 0:
            all_documents: GetResult = target_collection.get(include=['metadatas'], where={"chunk_idx": 0})
            all_metadatas: list[Mapping] = all_documents['metadatas']
            unique_documents: list[UniqueDocument] = []
            for metadata in all_metadatas:
                unique_documents.append(UniqueDocument(uuid=metadata['uuid'], source=metadata['source']))
            return unique_documents
        else:
            return []

    def delete_documents(self, uuids: List[str]):
        for uuid in uuids:
            self.collection.delete(where={'uuid': uuid})

    def query_llamaindex(self, query_text: str, response_mode: str = None,
                         collection_name: str = "default") -> str | None:
        if response_mode is None:
            response_mode = self.response_mode

        if collection_name is None or collection_name is self.default_collection_name or collection_name.strip() == "":
            collection_index: VectorStoreIndex = self.index
        else:
            collection_index: VectorStoreIndex = self.construct_index_for_collection(collection_name)

        query_engine: BaseQueryEngine = collection_index.as_query_engine(response_mode=response_mode,
                                                                         similarity_top_k=self.top_k)
        query_result: Response = query_engine.query(query_text)

        if response_mode == "no_text":
            query_response: str = query_result.get_formatted_sources(length=sys.maxsize)
        else:
            query_response: str = query_result.response

        return query_response

    def load_file_bytes(self, file_bytes: bytes, file_name: str, collection_name: str) -> None:
        active_collection: Collection = self.get_or_create_collection(collection_name)
        self.ingestor.load_file_bytes(file_bytes, file_name, active_collection)
