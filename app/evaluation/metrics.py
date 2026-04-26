from Levenshtein import distance


def calculate_term_accuracy(translation: str, key_terms: list[dict]) -> float:
    if not key_terms:
        return 0.0
    matched = 0
    lower = translation.lower()
    for term in key_terms:
        expected = term["expected_en_term"].lower()
        if expected in lower:
            matched += 1
    return matched / len(key_terms)


def calculate_edit_rate(translation: str, reference: str) -> float:
    max_len = max(len(translation), len(reference))
    if max_len == 0:
        return 0.0
    return distance(translation, reference) / max_len


def calculate_agent_accuracy(agent_judgments: list[bool], manual_labels: list[bool]) -> float:
    if not agent_judgments:
        return 0.0
    correct = sum(1 for a, m in zip(agent_judgments, manual_labels) if a == m)
    return correct / len(agent_judgments)
