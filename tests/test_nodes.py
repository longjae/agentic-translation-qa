from app.agents.nodes import decide_retry


def test_decide_retry() -> None:
    state = {"quality_pass": False, "attempt_count": 1, "max_attempts": 3}
    out = decide_retry(state)
    assert out["next_action"] == "retry"
