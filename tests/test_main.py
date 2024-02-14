from fastapi.testclient import TestClient

import main
from document_store import DocumentStore
from ingest import update_metadata
from main import app

TEST_COLLECTION_NAME = "test"


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


def test_list():
    if main.doc_store.collection.count() > 0:
        main.doc_store.client.delete_collection(TEST_COLLECTION_NAME)
        main.doc_store.create_collection(TEST_COLLECTION_NAME)

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
