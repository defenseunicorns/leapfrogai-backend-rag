# LeapfrogAI RAG Backend

> ***⚠️This repo is archived in favor of the LeapfrogAI monorepo: https://github.com/defenseunicorns/leapfrogai⚠️***

## Description

Stand alone Retrieval Augmented Generation Backend (RAG) for LeapfrogAI that used ChromaDB, LangChain, Llama Index, and FastAPI.

## Instructions

### Docker Container

#### Image Build and Run

For local image building and running.

``` bash
make docker-build

make docker-run # handles env file and db directory mount
```
