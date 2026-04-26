from datetime import datetime

from app.agents.graph import create_translation_graph
from app.core.config import settings
from app.rag.retriever import TermRetriever
from app.schemas.translation import AgentTranslationResponse


class AgentTranslationService:
    def __init__(self) -> None:
        self.graph = create_translation_graph()
        self.retriever = TermRetriever(
            glossary_dir=settings.data_path / "glossaries",
            index_dir=settings.data_path / "faiss_index",
        )

    async def translate(self, source_text: str, source_language: str, target_language: str) -> AgentTranslationResponse:
        request_id = f"tr_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        initial_state = {
            "source_text": source_text,
            "source_lang": source_language,
            "target_lang": target_language,
            "domain": "general",
            "key_terms": [],
            "retrieved_terms": [],
            "current_translation": "",
            "translation_context": "",
            "quality_score": 0.0,
            "quality_pass": False,
            "quality_issues": [],
            "quality_reasoning": "",
            "attempt_count": 0,
            "max_attempts": settings.max_attempts,
            "retry_history": [],
            "final_translation": "",
            "final_decision": "",
            "metrics": {},
            "retriever": self.retriever,
            "top_k": settings.top_k,
            "quality_pass_threshold": settings.quality_pass_threshold,
            "next_action": "retry",
        }

        state = await self.graph.ainvoke(initial_state)
        return AgentTranslationResponse(
            request_id=request_id,
            source_language=source_language,
            target_language=target_language,
            source_text=source_text,
            translated_text=state["final_translation"],
            model_info={
                "translation_model": settings.translation_model,
                "agent_model": settings.ollama_model,
                "embedding_model": settings.embedding_model,
                "vector_store": "faiss",
                "agent_framework": "langgraph",
            },
            analysis={"domain": state["domain"], "key_terms": state["key_terms"]},
            retrieval={"top_k": settings.top_k, "results": state["retrieved_terms"]},
            quality_assessment={
                "pass": state["quality_pass"],
                "score": state["quality_score"],
                "issues": state["quality_issues"],
                "reasoning_summary": state["quality_reasoning"],
            },
            retry={
                "attempt_count": state["attempt_count"],
                "max_allowed": state["max_attempts"],
                "history": state["retry_history"],
            },
            metrics=state["metrics"],
            final_decision={
                "accepted": state["quality_pass"],
                "acceptance_reason": "improved_after_rag_retry"
                if state["quality_pass"] and state["attempt_count"] > 1
                else ("accepted_first_try" if state["quality_pass"] else "max_attempts_reached"),
            },
        )
