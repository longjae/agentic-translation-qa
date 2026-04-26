from pydantic import BaseModel, Field


class TranslationRequest(BaseModel):
    source_text: str = Field(min_length=1)
    source_language: str = "ko"
    target_language: str = "en"

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_text": "이 약물은 간 기능이 저하된 환자에게 투여 시 주의가 필요하다.",
                "source_language": "ko",
                "target_language": "en",
            }
        }
    }


class ModelInfoBaseline(BaseModel):
    translation_model: str
    rag_applied: bool = False
    agent_applied: bool = False


class BaselineMetrics(BaseModel):
    latency_ms: int


class BaselineTranslationResponse(BaseModel):
    request_id: str
    mode: str = "baseline"
    source_language: str
    target_language: str
    source_text: str
    translated_text: str
    model_info: ModelInfoBaseline
    metrics: BaselineMetrics

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "tr_20260426_001",
                "mode": "baseline",
                "source_language": "ko",
                "target_language": "en",
                "source_text": "이 약물은 간 기능이 저하된 환자에게 투여 시 주의가 필요하다.",
                "translated_text": "This drug should be administered with caution in patients with impaired liver function.",
                "model_info": {
                    "translation_model": "Helsinki-NLP/opus-mt-ko-en",
                    "rag_applied": False,
                    "agent_applied": False,
                },
                "metrics": {"latency_ms": 842},
            }
        }
    }


class RetrievalItem(BaseModel):
    term: str
    recommended_translation: str
    score: float
    source: str


class AgentModelInfo(BaseModel):
    translation_model: str
    agent_model: str
    embedding_model: str
    vector_store: str = "faiss"
    agent_framework: str = "langgraph"


class AnalysisBlock(BaseModel):
    domain: str
    key_terms: list[str]


class RetrievalBlock(BaseModel):
    top_k: int
    results: list[RetrievalItem]


class QualityAssessment(BaseModel):
    pass_: bool = Field(alias="pass")
    score: float
    issues: list[str]
    reasoning_summary: str

    model_config = {"populate_by_name": True}


class RetryAttempt(BaseModel):
    attempt: int
    draft_translation: str
    quality_score: float
    decision: str
    failure_reasons: list[str] = []


class RetryBlock(BaseModel):
    attempt_count: int
    max_allowed: int
    history: list[RetryAttempt]


class AgentMetrics(BaseModel):
    total_latency_ms: int
    analysis_ms: int
    retrieval_ms: int
    translation_ms: int
    judgment_ms: int


class FinalDecision(BaseModel):
    accepted: bool
    acceptance_reason: str


class AgentTranslationResponse(BaseModel):
    request_id: str
    mode: str = "agent-rag"
    source_language: str
    target_language: str
    source_text: str
    translated_text: str
    model_info: AgentModelInfo
    analysis: AnalysisBlock
    retrieval: RetrievalBlock
    quality_assessment: QualityAssessment
    retry: RetryBlock
    metrics: AgentMetrics
    final_decision: FinalDecision

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "tr_20260426_002",
                "mode": "agent-rag",
                "source_language": "ko",
                "target_language": "en",
                "source_text": "본 계약은 당사자 일방의 중대한 과실이 있는 경우를 제외하고는 손해배상 책임을 제한한다.",
                "translated_text": "This agreement limits liability for damages except in cases of gross negligence by either party.",
                "model_info": {
                    "translation_model": "Helsinki-NLP/opus-mt-ko-en",
                    "agent_model": "gemma3:4b",
                    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "vector_store": "faiss",
                    "agent_framework": "langgraph",
                },
                "analysis": {
                    "domain": "legal",
                    "key_terms": ["중대한 과실", "손해배상 책임", "제한"],
                },
                "retrieval": {
                    "top_k": 3,
                    "results": [
                        {
                            "term": "중대한 과실",
                            "recommended_translation": "gross negligence",
                            "score": 0.94,
                            "source": "legal_glossary",
                        }
                    ],
                },
                "quality_assessment": {
                    "pass": True,
                    "score": 0.92,
                    "issues": [],
                    "reasoning_summary": "용어 정확도와 문맥 전달이 양호함.",
                },
                "retry": {
                    "attempt_count": 2,
                    "max_allowed": 3,
                    "history": [
                        {
                            "attempt": 1,
                            "draft_translation": "This contract limits liability...",
                            "quality_score": 0.51,
                            "decision": "retry",
                            "failure_reasons": ["term_mistranslation"],
                        },
                        {
                            "attempt": 2,
                            "draft_translation": "This agreement limits liability for damages except in cases of gross negligence by either party.",
                            "quality_score": 0.92,
                            "decision": "accept",
                            "failure_reasons": [],
                        },
                    ],
                },
                "metrics": {
                    "total_latency_ms": 4388,
                    "analysis_ms": 245,
                    "retrieval_ms": 55,
                    "translation_ms": 1890,
                    "judgment_ms": 2198,
                },
                "final_decision": {
                    "accepted": True,
                    "acceptance_reason": "improved_after_rag_retry",
                },
            }
        }
    }
