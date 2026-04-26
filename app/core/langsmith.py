import logging
import os

from app.core.config import settings

logger = logging.getLogger(__name__)


def setup_langsmith_tracing() -> None:
    if not settings.langsmith_tracing:
        logger.info("LangSmith tracing disabled")
        return

    if not settings.langsmith_api_key:
        logger.warning("LangSmith tracing enabled but LANGSMITH_API_KEY is missing")
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    logger.info("LangSmith tracing enabled for project=%s", settings.langsmith_project)
