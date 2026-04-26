from langgraph.graph import END, StateGraph

from app.agents.nodes import (
    analyze_sentence,
    decide_retry,
    finalize,
    judge_quality,
    retrieve_terms,
    translate,
)
from app.agents.state import TranslationState


def create_translation_graph():
    workflow = StateGraph(TranslationState)
    workflow.add_node("analyze", analyze_sentence)
    workflow.add_node("retrieve", retrieve_terms)
    workflow.add_node("translate", translate)
    workflow.add_node("judge", judge_quality)
    workflow.add_node("decide", decide_retry)
    workflow.add_node("finalize", finalize)

    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "retrieve")
    workflow.add_edge("retrieve", "translate")
    workflow.add_edge("translate", "judge")
    workflow.add_edge("judge", "decide")
    workflow.add_conditional_edges(
        "decide",
        lambda state: state["next_action"],
        {"retry": "retrieve", "finish": "finalize"},
    )
    workflow.add_edge("finalize", END)
    return workflow.compile()
