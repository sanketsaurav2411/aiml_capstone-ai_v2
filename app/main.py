from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import ALLOWED_ORIGINS, APP_ENV, APP_TITLE, APP_VERSION

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="Indian Customs & ITC-HS 2022 Hierarchical RAG and Stateful LangGraph Agent API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Internal-Api-Key"],
)

app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": f"{APP_TITLE} is running",
        "version": APP_VERSION,
        "environment": APP_ENV,
    }
