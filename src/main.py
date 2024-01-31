import os
import sys
from typing import List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from document_store import DocumentStore

path = os.getcwd()
path = os.path.join(path, ".env")
load_dotenv(path)

debug = False

app = FastAPI()

doc_store = DocumentStore()

origins = [
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
    input: str
    collection_name: str


@app.post("/upload/")
async def upload(file: UploadFile):
    try:
        debug("Received file: " + file.filename)
        contents: bytes = await file.read()
        doc_store.ingestor.load_file_bytes(contents, file.filename)
        debug("File loaded")
    except HTTPException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.detail,
        )
    except Exception as e:
        raise Exception(e)
    finally:
        await file.close()

    return {"filename": file.filename,
            "succeed": True}


@app.post("/query/")
def query(query_data: QueryModel):
    debug("Query received")
    outside_context = doc_store.query_llamaindex(query_data.input)
    debug("The returned context is: " + str(outside_context))
    return {"results": outside_context}


@app.post("/delete/")
def query(doc_ids: List[str] = Query(None)):
    if len(doc_ids) > 0:
        doc_store.delete_documents(doc_ids)


@app.get("/list/")
def query():
    return doc_store.get_all_documents()


@app.get("/health/", status_code=200)
def health():
    return {}


def debug(message):
    if debug:
        print(message)


if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) > 0 and args[0] == "debug":
        debug = True

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="debug")
