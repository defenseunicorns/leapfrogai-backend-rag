from enum import Enum
import os
import sys
from typing import List
import logging

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from document_store import DocumentStore

path = os.getcwd()
path = os.path.join(path, ".env")
load_dotenv(path)

debug = False

app = FastAPI()

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
        doc_store.load_file_bytes(contents, file.filename)
        logging.debug("File loaded")
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


def query_index(value: str, response_mode: str) -> QueryResponse:
    logging.debug("Query received")
    outside_context = doc_store.query_llamaindex(value, response_mode)
    logging.debug("The returned context is: " + str(outside_context))
    return QueryResponse(results=outside_context)


@app.post("/query/refined")
def query(query_data: QueryModel) -> QueryResponse:
    return query_index(query_data.input, "refine")


@app.post("/query/raw")
def query(query_data: QueryModel) -> QueryResponse:
    return query_index(query_data.input, "no_text")


@app.post("/delete/")
def query(doc_ids: List[str] = Query(None)) -> None:
    if len(doc_ids) > 0:
        doc_store.delete_documents(doc_ids)


@app.get("/list/")
def query() -> dict[str, str]:
    return doc_store.get_all_documents()


@app.get("/healthz/", status_code=200)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) > 0 and args[0] == "debug":
        debug = True

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="debug")
