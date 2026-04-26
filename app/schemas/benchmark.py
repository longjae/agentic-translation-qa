from pydantic import BaseModel


class BenchmarkRequest(BaseModel):
    dataset_name: str = "eval_set"
    sample_size: int = 40
    domain_filter: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "dataset_name": "eval_set",
                "sample_size": 40,
                "domain_filter": None,
            }
        }
    }


class SummaryMetrics(BaseModel):
    term_accuracy: float
    sentence_edit_rate: float
    avg_latency_ms: float
    agent_judgment_accuracy: float | None = None


class DeltaMetrics(BaseModel):
    term_accuracy_improvement: float
    sentence_edit_rate_improvement: float
    latency_overhead_ms: float


class BenchmarkSummary(BaseModel):
    baseline: SummaryMetrics
    agent_rag: SummaryMetrics
    delta: DeltaMetrics


class AgentStats(BaseModel):
    retry_distribution: dict[str, int]
    avg_retry_count: float
    pass_rate_by_attempt: dict[str, float]


class CaseItem(BaseModel):
    sample_id: str
    source_text: str
    reference: str
    baseline_output: str
    agent_rag_output: str
    improvement_reason: str | None = None
    failure_reason: str | None = None
    error_type: str | None = None


class BenchmarkResponse(BaseModel):
    benchmark_id: str
    dataset_info: dict
    summary: BenchmarkSummary
    agent_stats: AgentStats
    error_analysis: dict
    success_cases: list[CaseItem]
    failure_cases: list[CaseItem]

    model_config = {
        "json_schema_extra": {
            "example": {
                "benchmark_id": "bm_20260426_001",
                "dataset_info": {
                    "name": "custom_eval_set",
                    "source_dataset": "lemon-mint/korean_parallel_sentences_v1.1",
                    "sample_size": 40,
                    "domain_distribution": {"medical": 15, "legal": 13, "technical": 12},
                    "reference_included": True,
                },
                "summary": {
                    "baseline": {
                        "term_accuracy": 0.72,
                        "sentence_edit_rate": 0.38,
                        "avg_latency_ms": 811.0,
                        "agent_judgment_accuracy": None,
                    },
                    "agent_rag": {
                        "term_accuracy": 0.90,
                        "sentence_edit_rate": 0.17,
                        "avg_latency_ms": 3014.0,
                        "agent_judgment_accuracy": 0.84,
                    },
                    "delta": {
                        "term_accuracy_improvement": 0.18,
                        "sentence_edit_rate_improvement": -0.21,
                        "latency_overhead_ms": 2203.0,
                    },
                },
                "agent_stats": {
                    "retry_distribution": {"0_retry": 18, "1_retry": 15, "2_retry": 7},
                    "avg_retry_count": 0.73,
                    "pass_rate_by_attempt": {
                        "attempt_1": 0.45,
                        "attempt_2": 0.38,
                        "attempt_3": 0.17,
                    },
                },
                "error_analysis": {
                    "baseline": {"term_mistranslation": 11, "omission": 6, "context_error": 4, "register_mismatch": 3},
                    "agent_rag": {
                        "term_mistranslation": 4,
                        "omission": 3,
                        "context_error": 2,
                        "agent_misjudgment": 2,
                        "rag_retrieval_failure": 1,
                    },
                },
                "success_cases": [
                    {
                        "sample_id": "s12",
                        "source_text": "환자의 혈압이 급격히 상승하여 응급 조치가 필요합니다.",
                        "reference": "The patient's blood pressure rose sharply and emergency measures are needed.",
                        "baseline_output": "The patient's blood pressure suddenly increased and urgent action is needed.",
                        "agent_rag_output": "The patient's blood pressure rose sharply and emergency measures are needed.",
                        "improvement_reason": "핵심 의료 용어 정합성 개선",
                        "failure_reason": None,
                        "error_type": None,
                    }
                ],
                "failure_cases": [
                    {
                        "sample_id": "s27",
                        "source_text": "계약 해지 시 위약금은 계약 금액의 20%로 한다.",
                        "reference": "In case of contract termination, the penalty shall be 20% of the contract amount.",
                        "baseline_output": "When canceling the contract, the penalty is 20% of the contract amount.",
                        "agent_rag_output": "Upon contract cancellation, the penalty fee shall be 20% of the contract sum.",
                        "improvement_reason": None,
                        "failure_reason": "Agent 오판으로 부정확 번역 통과",
                        "error_type": "agent_misjudgment",
                    }
                ],
            }
        }
    }
