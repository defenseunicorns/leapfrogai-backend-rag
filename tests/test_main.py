from time import sleep

import pytest as pytest
from fastapi.testclient import TestClient

import main
from ingest import update_metadata
from main import app

TEST_COLLECTION_NAME = "test"


@pytest.fixture
def collection():
    try:
        main.doc_store.client.delete_collection(TEST_COLLECTION_NAME)
    except ValueError:
        "Collection does not exist, so it cannot be deleted."

    main.doc_store.collection = main.doc_store.client.get_or_create_collection(name=TEST_COLLECTION_NAME,
                                                                               embedding_function=main.doc_store.embeddings_function)
    main.doc_store.ingestor.collection = main.doc_store.collection


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
        response = client.get("/list/")
        assert response.status_code == 200
        assert len(response.json()) == 0

        test_collection = main.doc_store.client.get_collection(TEST_COLLECTION_NAME)
        test_collection.add(ids=["some-uuid"], embeddings=[15031, 12, 17566],
                            metadatas=update_metadata("test", "some-uuid", 0, {}))

        response = client.get("/list/")
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]["source"] == "test"


def test_delete(collection):
    with TestClient(app) as client:
        test_collection = main.doc_store.client.get_collection(TEST_COLLECTION_NAME)
        test_collection.add(ids=["some-uuid"], embeddings=[15031, 12, 17566],
                            metadatas=update_metadata("test", "some-uuid", 0, {}))

        response = client.get("/list/")
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]["source"] == "test"

        client.post("/delete/", params={"doc_ids": "some-uuid"})

        response = client.get("/list/")
        assert response.status_code == 200
        assert len(response.json()) == 0


def test_upload(collection):
    with open("tests/resources/lorem-ipsum.pdf", "rb") as f:
        _files = {'file': f}

        with TestClient(app) as client:
            response = client.get("/list/")
            assert response.status_code == 200
            assert len(response.json()) == 0

            response = client.post("/upload/", files=_files)
            assert response.status_code == 200

            test_collection = main.doc_store.collection
            while test_collection.count() == 0:
                sleep(1)

            response = client.get("/list/")
            assert response.status_code == 200
            assert len(response.json()) == 1


def test_query_raw(collection):
    with open("tests/resources/lorem-ipsum.pdf", "rb") as f:
        _files = {'file': f}
        with TestClient(app) as client:
            client.post("/upload/", files=_files)

            test_collection = main.doc_store.collection
            while test_collection.count() == 0:
                print("inside" + test_collection.count())
                sleep(1)
            print("outside" + test_collection.count())

            response = client.post("/query/raw", json={"input": "some input value",
                                                       "collection_name": TEST_COLLECTION_NAME})
            assert response.status_code == 200
            query_response: main.QueryResponse = response.json()
            assert "\xa0Lorem\xa0ipsum\xa0dolor" in query_response['results']
