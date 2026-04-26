from functools import lru_cache

from app.services.agent_translation import AgentTranslationService
from app.services.baseline_translation import BaselineTranslationService
from app.services.benchmark import BenchmarkService


@lru_cache
def get_baseline_service() -> BaselineTranslationService:
    return BaselineTranslationService()


@lru_cache
def get_agent_service() -> AgentTranslationService:
    return AgentTranslationService()


@lru_cache
def get_benchmark_service() -> BenchmarkService:
    return BenchmarkService(
        baseline_service=get_baseline_service(),
        agent_service=get_agent_service(),
    )
