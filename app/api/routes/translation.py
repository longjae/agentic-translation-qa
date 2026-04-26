from fastapi import APIRouter, Depends

from app.api.dependencies import get_agent_service, get_baseline_service
from app.schemas.translation import (
    AgentTranslationResponse,
    BaselineTranslationResponse,
    TranslationRequest,
)
from app.services.agent_translation import AgentTranslationService
from app.services.baseline_translation import BaselineTranslationService

router = APIRouter()


@router.post("/baseline", response_model=BaselineTranslationResponse)
def translate_baseline(
    request: TranslationRequest,
    service: BaselineTranslationService = Depends(get_baseline_service),
) -> BaselineTranslationResponse:
    return service.translate(
        source_text=request.source_text,
        source_language=request.source_language,
        target_language=request.target_language,
    )


@router.post("/agent-rag", response_model=AgentTranslationResponse)
async def translate_agent_rag(
    request: TranslationRequest,
    service: AgentTranslationService = Depends(get_agent_service),
) -> AgentTranslationResponse:
    return await service.translate(
        source_text=request.source_text,
        source_language=request.source_language,
        target_language=request.target_language,
    )
