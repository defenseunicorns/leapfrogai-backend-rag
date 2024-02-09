VERSION := $(shell git describe --abbrev=0 --tags 2> /dev/null )
ifeq ($(VERSION),)
  VERSION := latest
endif

ARCH := $(shell uname -m | sed s/aarch64/arm64/ | sed s/x86_64/amd64/)

create-venv:
	python -m venv .venv

activate-venv:
	source .venv/bin/activate

build-requirements:
	pip-compile -o requirements.txt pyproject.toml

build-requirements-dev:
	pip-compile --extra dev -o requirements-dev.txt pyproject.toml

requirements-dev:
	python -m pip install -r requirements-dev.txt

requirements:
	pip-sync requirements.txt requirements-dev.txt

docker-build:
	docker build -t ghcr.io/defenseunicorns/leapfrogai/rag:${VERSION}-${ARCH} . --build-arg ARCH=${ARCH}

docker-run:
	docker run -p 8000:8000 -v ~/.db/:/leapfrogai/db/ -d --env-file .env ghcr.io/defenseunicorns/leapfrogai/rag:${VERSION}-${ARCH}

docker-push:
	docker push ghcr.io/defenseunicorns/leapfrogai/rag:${VERSION}-${ARCH}