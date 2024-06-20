from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

class Health(BaseModel):
    name: str
    api_version: str

class Result(BaseModel):
    result: float

@app.get("/api/v1/health", response_model=Health, status_code=200)
async def health() -> dict:
    health = Health(name="API Demo", api_version="0.0.1")

    return health.model_dump()

# curl localhost:8000/square?num=2.718281
@app.get("/api/v1/square", response_model=Result, status_code=201)
async def square(num: float) -> dict:
    result = Result(result=num**2)

    return result.model_dump()
