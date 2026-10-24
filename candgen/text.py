import re

TOKEN_RE = re.compile(r"[а-яa-z0-9]+")
NON_ALNUM_RE = re.compile(r"[^а-яa-z0-9]")


def normalize(text) -> str:
    if not isinstance(text, str):
        return ""
    return text.lower().replace("ё", "е")


class WordAnalyzer:
    """Текст -> список основ слов. Объект, а не функция: так его можно сохранить в pickle
    вместе с CountVectorizer."""

    def __init__(self, stem: bool = True, max_tokens: int | None = None):
        self.stem = stem
        self.max_tokens = max_tokens
        self._cache: dict[str, str] = {}
        self._stemmer = None

    def _stem(self, token: str) -> str:
        if not self.stem:
            return token
        cached = self._cache.get(token)
        if cached is None:
            if self._stemmer is None:
                from nltk.stem.snowball import RussianStemmer

                self._stemmer = RussianStemmer()
            cached = self._stemmer.stem(token)
            self._cache[token] = cached
        return cached

    def __call__(self, text) -> list[str]:
        tokens = [self._stem(t) for t in TOKEN_RE.findall(normalize(text)) if len(t) > 1]
        return tokens[: self.max_tokens] if self.max_tokens else tokens

    def __getstate__(self):
        # Кэш и стеммер не сохраняем: кэш большой, стеммер пересоздаётся лениво.
        return {"stem": self.stem, "max_tokens": self.max_tokens}

    def __setstate__(self, state):
        self.__init__(**state)


class CharNgramAnalyzer:
    """Символьные n-граммы без пробелов и знаков: «1 с» и «1с» дают одно и то же.
    Устойчиво к опечаткам и словоформам, которые не ловит стеммер."""

    def __init__(self, n: int = 3):
        self.n = n

    def __call__(self, text) -> list[str]:
        s = NON_ALNUM_RE.sub("", normalize(text))
        if len(s) < self.n:
            return [s] if s else []
        return [s[i : i + self.n] for i in range(len(s) - self.n + 1)]
