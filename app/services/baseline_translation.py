import time
from datetime import datetime

from app.core.config import settings
from app.models.translator import get_translator
from app.schemas.translation import BaselineTranslationResponse


class BaselineTranslationService:
    def translate(self, source_text: str, source_language: str, target_language: str) -> BaselineTranslationResponse:
        started = time.perf_counter()
        translated = get_translator().translate(source_text)
        latency = int((time.perf_counter() - started) * 1000)
        request_id = f"tr_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        return BaselineTranslationResponse(
            request_id=request_id,
            source_language=source_language,
            target_language=target_language,
            source_text=source_text,
            translated_text=translated,
            model_info={
                "translation_model": settings.translation_model,
                "rag_applied": False,
                "agent_applied": False,
            },
            metrics={"latency_ms": latency},
        )
