import json
from pathlib import Path


def load_glossary_files(glossary_dir: Path) -> list[dict]:
    items: list[dict] = []
    for path in sorted(glossary_dir.glob("*_terms.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data:
            item = {
                "ko_term": row["ko_term"].strip(),
                "en_term": row["en_term"].strip(),
                "domain": row.get("domain", path.stem.replace("_terms", "")),
                "source": row.get("source", path.stem),
            }
            items.append(item)

    dedup: dict[str, dict] = {}
    for item in items:
        key = item["ko_term"]
        if key not in dedup:
            dedup[key] = item
    return list(dedup.values())
