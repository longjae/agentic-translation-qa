import json
from datetime import datetime

from app.core.config import settings
from app.evaluation.analyzer import classify_errors
from app.evaluation.metrics import calculate_agent_accuracy, calculate_edit_rate, calculate_term_accuracy
from app.schemas.benchmark import BenchmarkResponse


class BenchmarkService:
    def __init__(self, baseline_service, agent_service) -> None:
        self.baseline_service = baseline_service
        self.agent_service = agent_service

    async def run(self, dataset_name: str, sample_size: int, domain_filter: str | None = None) -> BenchmarkResponse:
        dataset_path = settings.data_path / "datasets" / f"{dataset_name}.json"
        samples = json.loads(dataset_path.read_text(encoding="utf-8"))
        if domain_filter:
            samples = [s for s in samples if s["domain"] == domain_filter]
        samples = samples[:sample_size]

        baseline_term, baseline_edit, baseline_lat = [], [], []
        agent_term, agent_edit, agent_lat = [], [], []
        agent_judgments, manual_labels = [], []
        retry_counts = []
        pass_rate_by_attempt = {"attempt_1": 0, "attempt_2": 0, "attempt_3": 0}
        pass_count_by_attempt = {"attempt_1": 0, "attempt_2": 0, "attempt_3": 0}
        error_types = [
            "term_mistranslation",
            "omission",
            "context_error",
            "agent_misjudgment",
            "rag_retrieval_failure",
        ]
        error_analysis = {
            "baseline": {k: 0 for k in error_types},
            "agent_rag": {k: 0 for k in error_types},
        }
        case_deltas: list[dict] = []

        for idx, sample in enumerate(samples, start=1):
            baseline = self.baseline_service.translate(sample["source_text"], "ko", "en")
            agent = await self.agent_service.translate(sample["source_text"], "ko", "en")
            key_terms = sample.get("key_terms", [])
            reference = sample["reference"]

            b_term = calculate_term_accuracy(baseline.translated_text, key_terms)
            a_term = calculate_term_accuracy(agent.translated_text, key_terms)
            b_edit = calculate_edit_rate(baseline.translated_text, reference)
            a_edit = calculate_edit_rate(agent.translated_text, reference)
            baseline_term.append(b_term)
            agent_term.append(a_term)
            baseline_edit.append(b_edit)
            agent_edit.append(a_edit)
            baseline_lat.append(baseline.metrics.latency_ms)
            agent_lat.append(agent.metrics.total_latency_ms)
            retry_counts.append(agent.retry.attempt_count - 1)

            if agent.retry.attempt_count <= 3:
                key = f"attempt_{agent.retry.attempt_count}"
                pass_count_by_attempt[key] += 1
                if agent.quality_assessment.pass_:
                    pass_rate_by_attempt[key] += 1

            agent_judgments.append(agent.quality_assessment.pass_)
            manual_pass = a_edit <= b_edit
            manual_labels.append(manual_pass)

            b_errors = classify_errors(reference, baseline.translated_text, key_terms)
            a_errors = classify_errors(reference, agent.translated_text, key_terms)
            if agent.quality_assessment.pass_ and not manual_pass:
                a_errors.append("agent_misjudgment")
            if not agent.retrieval.results:
                a_errors.append("rag_retrieval_failure")
            for tag in b_errors:
                if tag not in error_analysis["baseline"]:
                    error_analysis["baseline"][tag] = 0
                error_analysis["baseline"][tag] += 1
            for tag in a_errors:
                if tag not in error_analysis["agent_rag"]:
                    error_analysis["agent_rag"][tag] = 0
                error_analysis["agent_rag"][tag] += 1

            case = {
                "sample_id": sample.get("id", f"s{idx}"),
                "source_text": sample["source_text"],
                "reference": reference,
                "baseline_output": baseline.translated_text,
                "agent_rag_output": agent.translated_text,
            }
            case_deltas.append(
                {
                    **case,
                    "delta_edit": a_edit - b_edit,
                    "a_errors": a_errors,
                    "manual_pass": manual_pass,
                }
            )

        for k, v in pass_rate_by_attempt.items():
            total = pass_count_by_attempt[k]
            pass_rate_by_attempt[k] = (v / total) if total else 0.0

        # 성공/실패 사례는 전체 후보에서 우선순위 기준으로 3건씩 선정
        sorted_cases = sorted(case_deltas, key=lambda x: x["delta_edit"])
        success_candidates = [c for c in sorted_cases if c["delta_edit"] <= 0]
        failure_candidates = [c for c in sorted(case_deltas, key=lambda x: x["delta_edit"], reverse=True) if c["delta_edit"] > 0]

        success_cases = [
            {
                "sample_id": c["sample_id"],
                "source_text": c["source_text"],
                "reference": c["reference"],
                "baseline_output": c["baseline_output"],
                "agent_rag_output": c["agent_rag_output"],
                "improvement_reason": "lower_or_equal_edit_rate_with_agent_rag",
            }
            for c in success_candidates[:3]
        ]
        failure_cases = [
            {
                "sample_id": c["sample_id"],
                "source_text": c["source_text"],
                "reference": c["reference"],
                "baseline_output": c["baseline_output"],
                "agent_rag_output": c["agent_rag_output"],
                "failure_reason": "higher_edit_rate_or_judgment_issue_with_agent_rag",
                "error_type": c["a_errors"][0] if c["a_errors"] else "context_error",
            }
            for c in failure_candidates[:3]
        ]

        avg = lambda arr: sum(arr) / len(arr) if arr else 0.0
        response = BenchmarkResponse(
            benchmark_id=f"bm_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            dataset_info={
                "name": "custom_eval_set",
                "source_dataset": "lemon-mint/korean_parallel_sentences_v1.1",
                "sample_size": len(samples),
                "domain_distribution": {
                    "medical": sum(1 for s in samples if s["domain"] == "medical"),
                    "legal": sum(1 for s in samples if s["domain"] == "legal"),
                    "technical": sum(1 for s in samples if s["domain"] == "technical"),
                },
                "reference_included": True,
            },
            summary={
                "baseline": {
                    "term_accuracy": avg(baseline_term),
                    "sentence_edit_rate": avg(baseline_edit),
                    "avg_latency_ms": avg(baseline_lat),
                    "agent_judgment_accuracy": None,
                },
                "agent_rag": {
                    "term_accuracy": avg(agent_term),
                    "sentence_edit_rate": avg(agent_edit),
                    "avg_latency_ms": avg(agent_lat),
                    "agent_judgment_accuracy": calculate_agent_accuracy(agent_judgments, manual_labels),
                },
                "delta": {
                    "term_accuracy_improvement": avg(agent_term) - avg(baseline_term),
                    "sentence_edit_rate_improvement": avg(agent_edit) - avg(baseline_edit),
                    "latency_overhead_ms": avg(agent_lat) - avg(baseline_lat),
                },
            },
            agent_stats={
                "retry_distribution": {
                    "0_retry": sum(1 for x in retry_counts if x == 0),
                    "1_retry": sum(1 for x in retry_counts if x == 1),
                    "2_retry": sum(1 for x in retry_counts if x >= 2),
                },
                "avg_retry_count": avg(retry_counts),
                "pass_rate_by_attempt": pass_rate_by_attempt,
            },
            error_analysis=error_analysis,
            success_cases=success_cases,
            failure_cases=failure_cases,
        )
        settings.outputs_path.mkdir(parents=True, exist_ok=True)
        out_file = settings.outputs_path / f"{response.benchmark_id}.json"
        out_file.write_text(response.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
        return response
