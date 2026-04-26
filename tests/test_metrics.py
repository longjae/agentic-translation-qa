from app.evaluation.metrics import calculate_agent_accuracy, calculate_edit_rate, calculate_term_accuracy


def test_calculate_term_accuracy() -> None:
    key_terms = [{"expected_en_term": "blood pressure"}, {"expected_en_term": "emergency measures"}]
    score = calculate_term_accuracy("blood pressure and emergency measures are required", key_terms)
    assert score == 1.0


def test_calculate_edit_rate() -> None:
    assert calculate_edit_rate("abc", "abc") == 0.0


def test_calculate_agent_accuracy() -> None:
    assert calculate_agent_accuracy([True, False, True], [True, False, False]) == 2 / 3
