import logging
import os
import sys
import threading
from typing import List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from document_store import DocumentStore, UniqueDocument

path = os.getcwd()
path = os.path.join(path, ".env")
load_dotenv(path)

debug = False

prefix: str = os.environ['PREFIX'] or ""
app = FastAPI(root_path=prefix)

doc_store = DocumentStore()

origins: list[str] = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryModel(BaseModel):
    input: str = Field(default=None, examples=["List some key points from the documents."])
    collection_name: str = Field(default="default")


class UploadResponse(BaseModel):
    filename: str
    succeed: bool


class QueryResponse(BaseModel):
    results: str


class HealthResponse(BaseModel):
    status: str


@app.post("/upload/")
async def upload(file: UploadFile) -> UploadResponse:
    try:
        logging.debug("Received file: " + file.filename)
        contents: bytes = await file.read()
        thread = threading.Thread(target=doc_store.load_file_bytes, args=(contents, file.filename))
        thread.start()
        logging.debug("File load started")
    except HTTPException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.detail,
        )
    except Exception as e:
        raise Exception(e)
    finally:
        await file.close()

    return UploadResponse(filename=file.filename, succeed=True)


def query_index(value: str, response_mode: str, collection_name: str) -> QueryResponse:
    logging.debug("Query received")
    outside_context = doc_store.query_llamaindex(value, response_mode, collection_name)
    logging.debug("The returned context is: " + str(outside_context))
    return QueryResponse(results=outside_context)


@app.post("/query/refined")
def query(query_data: QueryModel) -> QueryResponse:
    return query_index(query_data.input, "refine", query_data.collection_name)


@app.post("/query/raw")
def query(query_data: QueryModel) -> QueryResponse:
    return query_index(query_data.input, "no_text", query_data.collection_name)


@app.post("/delete/")
def query(doc_ids: List[str] = Query(None)) -> None:
    if len(doc_ids) > 0:
        doc_store.delete_documents(doc_ids)


@app.get("/list/")
def query() -> list[UniqueDocument]:
    return doc_store.get_all_documents()


@app.get("/healthz", status_code=200)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) > 0 and args[0] == "debug":
        debug = True
        
    logging.basicConfig(level=logging.DEBUG)

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="debug")
