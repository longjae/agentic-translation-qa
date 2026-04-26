import json
import re
from functools import lru_cache

from langchain_ollama import ChatOllama

from app.core.config import settings


def parse_json_safe(text: str) -> dict:
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # LLM이 trailing comma/주석 유사 텍스트를 섞는 경우 최소 정규화
                normalized = re.sub(r",\s*([}\]])", r"\1", candidate)
                try:
                    return json.loads(normalized)
                except json.JSONDecodeError:
                    pass
        # 파싱 실패 시 빈 dict를 반환하고 상위 로직의 기본값으로 진행
        return {}


@lru_cache
def get_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
        timeout=settings.request_timeout_s,
    )
