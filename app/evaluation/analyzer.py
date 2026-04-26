def classify_errors(reference: str, output: str, key_terms: list[dict]) -> list[str]:
    errors: list[str] = []
    lower_output = output.lower()
    for term in key_terms:
        if term["expected_en_term"].lower() not in lower_output:
            errors.append("term_mistranslation")
            break
    if len(output.split()) < max(1, int(len(reference.split()) * 0.7)):
        errors.append("omission")
    if not errors:
        if output.strip().lower() != reference.strip().lower():
            errors.append("context_error")
    return errors
