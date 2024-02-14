from fastapi.testclient import TestClient
from main import app


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
    with TestClient(app) as client:
        response = client.get("/list/")
        assert response.status_code == 200
        assert len(response.json()) > 0
