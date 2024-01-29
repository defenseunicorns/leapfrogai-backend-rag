import os
import sys

import uvicorn
from fastapi import FastAPI, UploadFile, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from document_store import DocumentStore

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
        contents = await file.read()
        file_folder = "files/"
        full_file_path = file_folder + file.filename
        # TODO: Change this to be in memory
        if not os.path.isfile(full_file_path):
            if not os.path.exists(file_folder):
                os.mkdir(file_folder)
            new_file = open(full_file_path, "xb")
            new_file.write(contents)
        doc_store.ingestor.process_file(full_file_path)
        os.remove(full_file_path)
    except HTTPException as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='There was an error uploading the file',
        )
    except Exception as e:
        print(e)
        raise Exception(e)
    finally:
        await file.close()

    return {"filename": file.filename,
            "succeed": True}


@app.post("/query/")
def query(query_data: QueryModel):
    debug("Query received")
    outside_context = doc_store.query_langchain(query_data.input)
    debug("The returned context is: " + str(outside_context))
    return {"results": outside_context}


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
