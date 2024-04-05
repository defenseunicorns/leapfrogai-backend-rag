import logging
import os
import time
from time import sleep

from chromadb import Collection, GetResult
import pytest as pytest
from fastapi.testclient import TestClient
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from llama_index import ServiceContext
from tenacity import retry, wait_random_exponential, stop_after_attempt, stop_after_delay, wait_fixed

import main
from embedding_functions import PassThroughEmbeddingsFunction
from ingest import update_metadata
from main import app

TEST_COLLECTION_NAME = "test"


# Helper Functions
def http_get_list_from_collection(client: TestClient, collection_name: str = TEST_COLLECTION_NAME):
    return client.get("/list/", params={"collection_name": collection_name})


def http_post_query_raw_from_collection(client: TestClient, query: str, collection_name: str = TEST_COLLECTION_NAME):
    return client.post("/query/raw", json={"input": query, "collection_name": collection_name})


def http_post_delete_ids_from_collection(client: TestClient, uuids: str, collection_name: str = TEST_COLLECTION_NAME):
    client.post("/delete/", params={"doc_ids": uuids, "collection_name": collection_name})


def http_upload_files_to_collection(client: TestClient, files=None, collection_name: str = TEST_COLLECTION_NAME):
    return client.post("/upload/", files=files, params={"collection_name": collection_name})


def add_files_to_collection(ids: [str], embeddings: [int], metadatas: dict,
                            collection_name: str = TEST_COLLECTION_NAME):
    test_collection: Collection = main.doc_store.get_or_create_collection(collection_name)
    test_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)


def get_files_from_collection(include: [str], where: dict,
                              collection_name: str = TEST_COLLECTION_NAME) -> GetResult:
    test_collection: Collection = main.doc_store.get_or_create_collection(collection_name)
    return test_collection.get(include=include, where=where)


@pytest.fixture
def collection():
    main.doc_store.embeddings_model_name = os.environ.get("EMBEDDING_MODEL_NAME") or "hkunlp/instructor-xl"
    main.doc_store.embeddings = SentenceTransformerEmbeddings(
        model_name=main.doc_store.embeddings_model_name,
        model_kwargs={'device': 'cpu'},
    )
    main.doc_store.embeddings_function = PassThroughEmbeddingsFunction(
        model_name=main.doc_store.embeddings_model_name,
        embeddings=main.doc_store.embeddings)
    main.doc_store.chroma_db = Chroma(embedding_function=main.doc_store.embeddings,
                                      collection_name=TEST_COLLECTION_NAME,
                                      client=main.doc_store.client)
    main.doc_store.service_context = ServiceContext.from_defaults(embed_model=main.doc_store.embeddings,
                                                                  llm=main.doc_store.llm,
                                                                  context_window=main.doc_store.context_window,
                                                                  num_output=main.doc_store.max_output,
                                                                  chunk_size=main.doc_store.chunk_size,
                                                                  chunk_overlap=main.doc_store.overlap_size)

    try:
        main.doc_store.client.delete_collection(TEST_COLLECTION_NAME)
    except ValueError:
        logging.debug("Collection does not exist, so it cannot be deleted.")


def test_routes():
    expected_routes = {
        "/docs": ['GET', 'HEAD'],
        "/upload/": ['POST'],
        "/query/refined": ['POST'],
        "/query/raw": ['POST'],
        "/delete/": ['POST'],
        "/list/": ['GET'],
        "/healthz": ['GET'],
    }

    actual_routes = app.routes
    for route in actual_routes:
        if hasattr(route, "path") and route.path in expected_routes:
            assert route.methods == set(expected_routes[route.path])
            del expected_routes[route.path]

    assert len(expected_routes) == 0


def test_healthz():
    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_list(collection):
    with TestClient(app) as client:
        response = http_get_list_from_collection(client)
        assert response.status_code == 200
        assert len(response.json()) == 0

        add_files_to_collection(["some-uuid"], [15031, 12, 17566],
                                update_metadata("test", "some-uuid", 0, {}))

        response = http_get_list_from_collection(client, TEST_COLLECTION_NAME)
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]["source"] == "test"


def test_delete(collection):
    with TestClient(app) as client:
        add_files_to_collection(["some-uuid"], [15031, 12, 17566],
                                update_metadata("test", "some-uuid", 0, {}))

        response = http_get_list_from_collection(client)
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]["source"] == "test"

        http_post_delete_ids_from_collection(client, "some-uuid")

        response = http_get_list_from_collection(client)
        assert response.status_code == 200
        assert len(response.json()) == 0


def test_upload(collection):
    with open("tests/resources/lorem-ipsum.pdf", "rb") as f:
        files = {'file': f}

        with TestClient(app) as client:
            response = http_get_list_from_collection(client)
            assert response.status_code == 200
            assert len(response.json()) == 0

            response = http_upload_files_to_collection(client, files)
            assert response.status_code == 200

            sleep(35)

            response = http_get_list_from_collection(client)
            assert response.status_code == 200
            assert len(response.json()) == 1


def test_query_raw(collection):
    with open("tests/resources/lorem-ipsum.pdf", "rb") as f:
        files = {'file': f}
        with TestClient(app) as client:
            http_upload_files_to_collection(client, files)

            sleep(35)

            doc = get_files_from_collection(['metadatas'], {"chunk_idx": 0})
            assert "lorem-ipsum.pdf" in doc['metadatas'][0]['source']

            response = http_post_query_raw_from_collection(client, "lorem ipsum dolor")
            assert response.status_code == 200
            query_response: main.QueryResponse = response.json()
            assert "\xa0Lorem\xa0ipsum\xa0dolor" in query_response['results']
