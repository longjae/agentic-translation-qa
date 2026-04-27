# Agentic Translation QA

FastAPI + LangGraph + 로컬 모델(Ollama/MarianMT/FAISS) 기반 번역 품질 개선 시스템입니다.

## 1. 환경

- Python 3.11+
- 패키지 매니저: `uv`

## 2. 설치

```bash
uv venv
uv sync
```

## 3. 모델 준비

```bash
ollama pull gemma3:4b
```

MarianMT, SentenceTransformer는 최초 실행 시 캐시됩니다.

## 4. 서버 실행

```bash
uv run app/main.py
```

## 5. 테스트

```bash
uv run pytest
```

## 6. API

- `POST /translate/baseline`
- `POST /translate/agent-rag`
- `POST /benchmark`

예시 요청:

```json
{
  "source_text": "이 약물은 간 기능이 저하된 환자에게 투여 시 주의가 필요하다.",
  "source_language": "ko",
  "target_language": "en"
}
```

[👉 report.md 바로가기](report.md)