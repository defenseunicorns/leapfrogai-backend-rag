from pydantic import BaseModel, Field

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
