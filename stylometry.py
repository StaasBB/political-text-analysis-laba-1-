from pathlib import Path
import re
from statistics import mean # avg


# Разбиваем на предложения по . ! ? … (с учётом многоточий).
# Знак остаётся в предыдущем предложении.
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")

# Слово — последовательность букв (кириллица + латиница),
# допускаем внутренний дефис: «кто-то», «по-моему».
WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:-[А-Яа-яЁёA-Za-z]+)*")

# Знаки пунктуации, которые считаем в метрике 3.
PUNCT_RE = re.compile(r"[.,!?;:…—–\-«»\"'()\[\]{}]")


#---------утилиты

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


# ---------метрики

def metric_avg_sentence_len_chars(clean_text: str) -> float:
    """
    Средняя длина предложения в символах.
    Предложения выделяются по [.!?…] + пробел.
    """
    parts = SENT_SPLIT.split(clean_text)
    lengths = [len(p.strip()) for p in parts if len(p.strip())]
    return mean(lengths) if lengths else 0.0


def metric_avg_word_len(clean_text: str) -> float:
    """
    Средняя длина слов в символах.
    """
    words = WORD_RE.findall(clean_text)
    return mean(len(w) for w in words) if words else 0.0


def metric_punct_count(clean_text: str) -> int:
    """
    Общее количество знаков пунктуации в тексте.
    Считается по всем вхождениям.
    """
    return len(PUNCT_RE.findall(clean_text))

def metric_total_chars(text: str) -> int:
    """Всего символов (включая пробелы и пунктуацию)."""
    return len(text)


def metric_total_letters(text: str) -> int:
    """Всего букв (isalpha — корректно для Unicode)."""
    return sum(1 for ch in text if ch.isalpha())


def metric_total_words(text: str) -> int:
    """Всего слов с повторами."""
    return len(get_words(text))


def metric_total_sentences(text: str) -> int:
    """Всего предложений (после фильтра ≥ 5 символов)."""
    return len(split_sentences(text))


def metric_unique_words(text: str) -> int:
    """Уникальных слов (регистрозависимо)."""
    return len(set(get_words(text)))


def metric_ttr(text: str) -> float:
    """Type-Token Ratio = уникальные / всего. Зависит от длины текста."""
    words = get_words(text)
    return len(set(words)) / len(words) if words else 0.0


def metric_avg_sent_len_words(text: str) -> float:
    """Средняя длина предложения в словах."""
    sentences = split_sentences(text)
    return mean(len(get_words(s)) for s in sentences) if sentences else 0.0


def metric_capital_ratio(text: str) -> float:
    """Доля заглавных букв среди всех букв."""
    letters = [ch for ch in text if ch.isalpha()]
    return sum(1 for ch in letters if ch.isupper()) / len(letters) if letters else 0.0


def metric_digit_count(text: str) -> int:
    """Количество цифр."""
    return sum(1 for ch in text if ch.isdigit())

#сборка вектора для одного текста

def extract_features(clean_path: Path) -> dict:
    """Собирает все метрики в один вектор-словарь."""
    text = load_clean(clean_path)
    return {
        "avg_sentence_len_chars": metric_avg_sentence_len_chars(text),
        "avg_word_len":           metric_avg_word_len(text),
        "punct_count":            metric_punct_count(text),
        "total_chars":            metric_total_chars(text),
        "total_letters":          metric_total_letters(text),
        "total_words":            metric_total_words(text),
        "total_sentences":        metric_total_sentences(text),
        "unique_words":           metric_unique_words(text),
        "ttr":                    metric_ttr(text),
        "avg_sent_len_words":     metric_avg_sent_len_words(text),
        "capital_ratio":          metric_capital_ratio(text),
        "digit_count":            metric_digit_count(text),
    }