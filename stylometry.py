"""
Стилометрические метрики для корпуса political_corpus.
Реализовано:
  1. avg_sentence_len_chars — средняя длина предложения в символах
  2. avg_word_len           — средняя длина слова в символах
  3. avg_sent_len_words     — средняя длина предложения в словах
  4. ttr                    — лексическое разнообразие (уникальные / всего)
  5. capital_ratio          — доля заглавных среди всех букв
  6. punct_ratio            — доля пунктуации (знаки / все символы)
  7. digit_ratio            — доля цифр (цифры / все символы)
  8. comma_ratio            — доля запятых среди знаков пунктуации
"""

from pathlib import Path
import re
from statistics import mean


# ---------- регулярки

# Разбиение на предложения: . ! ? … + пробел.
# Lookbehind (?<=...) сохраняет знак в конце предыдущего предложения.
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")

# Слова: буквы (кириллица/латиница), допускается внутренний дефис.
WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:-[А-Яа-яЁёA-Za-z]+)*")

# Знаки пунктуации — для метрики punct_ratio.
PUNCT_RE = re.compile(r"[.,!?;:…—–\-«»\"'()\[\]{}]")

# Отдельно — запятые. Для метрики comma_ratio:
COMMA_RE = re.compile(r",")


# ---------- утилиты

def load_clean(path: Path) -> str:
    """Читает очищенный текст."""
    return path.read_text(encoding="utf-8")


def split_sentences(text: str) -> list[str]:
    """Режет текст на предложения, отбрасывает фрагменты < 5 символов."""
    parts = SENT_SPLIT.split(text)
    return [p.strip() for p in parts if len(p.strip()) >= 5]


def get_words(text: str) -> list[str]:
    """Извлекает слова с повторами (не set)."""
    return WORD_RE.findall(text)


# ---------- метрики

def metric_avg_sentence_len_chars(text: str) -> float:
    """Средняя длина предложения в символах."""
    parts = SENT_SPLIT.split(text)
    lengths = [len(p.strip()) for p in parts if len(p.strip())]
    return mean(lengths) if lengths else 0.0


def metric_avg_word_len(text: str) -> float:
    """Средняя длина слова в символах."""
    words = get_words(text)
    return mean(len(w) for w in words) if words else 0.0


def metric_avg_sent_len_words(text: str) -> float:
    """Средняя длина предложения в словах."""
    sentences = split_sentences(text)
    return mean(len(get_words(s)) for s in sentences) if sentences else 0.0


def metric_ttr(text: str) -> float:
    """
    Type-Token Ratio = уникальные / всего.
    """
    words = get_words(text)
    return len(set(words)) / len(words) if words else 0.0


def metric_capital_ratio(text: str) -> float:
    """Доля заглавных букв среди всех букв."""
    letters = [ch for ch in text if ch.isalpha()]
    return sum(1 for ch in letters if ch.isupper()) / len(letters) if letters else 0.0


def metric_punct_ratio(text: str) -> float:
    """
    Доля знаков пунктуации среди всех символов.
    Нормировка: punct_count / total_chars.
    """
    if not text:
        return 0.0
    return len(PUNCT_RE.findall(text)) / len(text)


def metric_digit_ratio(text: str) -> float:
    """
    Доля цифр среди всех символов.
    Нормировка: digit_count / total_chars.
    """
    if not text:
        return 0.0
    return sum(1 for ch in text if ch.isdigit()) / len(text)


def metric_comma_ratio(text: str) -> float:
    """
    Доля запятых среди всех знаков пунктуации.
    Нормировка на общее число знаков пунктуации — чтобы метрика
    не зависела от длины текста.
    """
    punct_total = len(PUNCT_RE.findall(text))
    if punct_total == 0:
        return 0.0
    return len(COMMA_RE.findall(text)) / punct_total


# ---------- сборка вектора

def extract_features(clean_path: Path) -> dict:
    """
    Собирает все метрики в один вектор-словарь.
    Все 8 метрик нормированы — не зависят от длины текста.
    """
    text = load_clean(clean_path)

    return {
        "avg_sentence_len_chars": metric_avg_sentence_len_chars(text),
        "avg_word_len":           metric_avg_word_len(text),
        "avg_sent_len_words":     metric_avg_sent_len_words(text),
        "ttr":                    metric_ttr(text),
        "capital_ratio":          metric_capital_ratio(text),
        "punct_ratio":            metric_punct_ratio(text),
        "digit_ratio":            metric_digit_ratio(text),
        "comma_ratio":            metric_comma_ratio(text),
    }