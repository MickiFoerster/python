from typing import Any

from fastapi import FastAPI, Request, APIRouter
from fastapi.responses import RedirectResponse
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from loguru import logger


class Health(BaseModel):
    name: str
    api_version: str

class Result(BaseModel):
    result: float

class Settings:
    API_ENDPOINT_PREFIX: str = "/api/v1"

    #LOGGING_LEVEL: int = logging.INFO

    TITLE: str = "Project Title"


settings = Settings()

app = FastAPI(
        title=settings.TITLE,
        openapi_url=f"{settings.API_ENDPOINT_PREFIX}/openapi.json"
)

#app.add_middleware(
#        CORSMiddleware, allow_origins=["http://localhost:8001"], allow_methods=["*"], allow_headers=["*"]
#)

root_router = APIRouter()
api_router = APIRouter(prefix=settings.API_ENDPOINT_PREFIX)

@root_router.get("/")
async def index(request: Request) -> Any:
    return RedirectResponse(url="/docs")


@root_router.get("/health", response_model=Health, status_code=200)
async def health() -> dict:
    health = Health(name="API Demo", api_version="0.0.1")

    return health.model_dump()


# curl localhost:8000/square?num=2.718281
@api_router.get("/square", response_model=Result, status_code=200)
async def square(num: float) -> dict:
    logger.info("endpoint SQUARE was called")
    result = Result(result=num**2)

    return result.model_dump()

@api_router.post("/create", response_model=Result, status_code=201)
async def create(num: float) -> dict:
    logger.info("endpoint CREATE was called")
    result = Result(result=num**2)

    return result.model_dump()


app.include_router(root_router)
app.include_router(api_router)



if __name__ == "__main__":
    logger.debug("server starts")

    import uvicorn

    uvicorn.run(app, host="localhost", port=8001, log_level="debug")
