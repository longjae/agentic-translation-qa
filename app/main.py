from fastapi import FastAPI
import uvicorn

from app.api.routes.benchmark import router as benchmark_router
from app.api.routes.translation import router as translation_router
from app.core.config import settings
from app.core.langsmith import setup_langsmith_tracing
from app.core.logging import setup_logging

setup_logging()
setup_langsmith_tracing()

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.include_router(translation_router, prefix="/translate", tags=["translation"])
app.include_router(benchmark_router, prefix="/benchmark", tags=["benchmark"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
