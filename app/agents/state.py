import operator
from typing import Annotated, TypedDict


class TranslationState(TypedDict):
    source_text: str
    source_lang: str
    target_lang: str
    domain: str
    key_terms: list[str]
    retrieved_terms: list[dict]
    current_translation: str
    translation_context: str
    quality_score: float
    quality_pass: bool
    quality_issues: list[str]
    quality_reasoning: str
    attempt_count: int
    max_attempts: int
    retry_history: Annotated[list[dict], operator.add]
    final_translation: str
    final_decision: str
    metrics: dict
    retriever: object
    top_k: int
    quality_pass_threshold: float
    next_action: str
