import time

from app.models.llm import get_llm, parse_json_safe
from app.models.translator import get_translator
from app.rag.retriever import TermRetriever


def _format_terms(terms: list[dict]) -> str:
    return "\n".join(
        [f"- {t['term']} => {t['recommended_translation']} (score={t['score']:.2f})" for t in terms]
    )


async def analyze_sentence(state):
    started = time.perf_counter()
    llm = get_llm()
    prompt = f"""다음 한국어 문장을 분석하고 JSON만 반환하세요.
문장: "{state['source_text']}"
키: domain(medical/legal/technical/general), key_terms(최대 5개)
"""
    response = await llm.ainvoke(prompt)
    data = parse_json_safe(response.content)
    metrics = dict(state["metrics"])
    metrics["analysis_ms"] = int((time.perf_counter() - started) * 1000)
    return {
        **state,
        "domain": data.get("domain", "general"),
        "key_terms": data.get("key_terms", [])[:5],
        "metrics": metrics,
    }


async def retrieve_terms(state):
    started = time.perf_counter()
    retriever: TermRetriever = state["retriever"]
    retrieved = retriever.retrieve(state.get("key_terms", []), top_k=state["top_k"])
    metrics = dict(state["metrics"])
    metrics["retrieval_ms"] = int((time.perf_counter() - started) * 1000)
    return {**state, "retrieved_terms": retrieved, "metrics": metrics}


async def translate(state):
    started = time.perf_counter()
    translator = get_translator()
    terms = state.get("retrieved_terms", [])
    context = _format_terms(terms) if terms else ""
    translated = translator.translate(state["source_text"])
    for item in terms:
        if item["term"] in state["source_text"] and item["recommended_translation"] not in translated:
            translated = f"{translated} ({item['recommended_translation']})"
    metrics = dict(state["metrics"])
    metrics["translation_ms"] = int((time.perf_counter() - started) * 1000)
    return {
        **state,
        "translation_context": context,
        "current_translation": translated,
        "attempt_count": state["attempt_count"] + 1,
        "metrics": metrics,
    }


async def judge_quality(state):
    started = time.perf_counter()
    llm = get_llm()
    prompt = f"""다음 번역 품질을 평가하고 JSON만 반환하세요.
원문: {state['source_text']}
번역문: {state['current_translation']}
용어집: {_format_terms(state.get('retrieved_terms', []))}
키: pass(boolean), score(0~1), issues(list), reasoning(string)
"""
    response = await llm.ainvoke(prompt)
    data = parse_json_safe(response.content)
    passed = bool(data.get("pass", False))
    score = float(data.get("score", 0.0))
    issues = data.get("issues", [])
    reasoning = data.get("reasoning", "")
    attempt_record = {
        "attempt": state["attempt_count"],
        "draft_translation": state["current_translation"],
        "quality_score": score,
        "decision": "accept" if passed else "retry",
        "failure_reasons": issues,
    }
    metrics = dict(state["metrics"])
    metrics["judgment_ms"] = int((time.perf_counter() - started) * 1000)
    return {
        **state,
        "quality_score": score,
        "quality_pass": passed or score >= state["quality_pass_threshold"],
        "quality_issues": issues,
        "quality_reasoning": reasoning,
        "retry_history": [attempt_record],
        "metrics": metrics,
    }


def decide_retry(state):
    should_retry = (not state["quality_pass"]) and (state["attempt_count"] < state["max_attempts"])
    return {**state, "next_action": "retry" if should_retry else "finish"}


def finalize(state):
    total_ms = sum(
        state["metrics"].get(k, 0) for k in ["analysis_ms", "retrieval_ms", "translation_ms", "judgment_ms"]
    )
    metrics = dict(state["metrics"])
    metrics["total_latency_ms"] = total_ms
    return {
        **state,
        "final_translation": state["current_translation"],
        "final_decision": "accepted" if state["quality_pass"] else "max_attempts_reached",
        "metrics": metrics,
    }
