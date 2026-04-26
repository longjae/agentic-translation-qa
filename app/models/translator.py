from functools import lru_cache

from transformers import MarianMTModel, MarianTokenizer

from app.core.config import settings


class MarianTranslator:
    def __init__(self, model_name: str) -> None:
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate(self, text: str) -> str:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True)
        generated = self.model.generate(**tokens, max_length=256)
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)


@lru_cache
def get_translator() -> MarianTranslator:
    return MarianTranslator(settings.translation_model)
