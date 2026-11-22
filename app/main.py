from fastapi import FastAPI

from app.api.routes import router
from app.config import APP_ENV, APP_TITLE, APP_VERSION

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="RAG API for answering user questions using a vector database.",
)

app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": f"{APP_TITLE} is running",
        "version": APP_VERSION,
        "environment": APP_ENV,
    }
