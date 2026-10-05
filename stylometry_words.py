"""
Метрики N-грамм СЛОВ (лемм) на основе processed-файлов.

Реализовано:
  N-граммы лемм длины 1, 2, 3 — частоты топ-N по корпусу.

"""

from pathlib import Path
import ast
from collections import Counter, defaultdict


# ---------- чтение processed

def load_processed(path: Path) -> list[dict]:
    """Читает processed-файл — это str(list) с токенами Stanza."""
    text = path.read_text(encoding="utf-8")
    data = ast.literal_eval(text)
    if not isinstance(data, list):
        raise ValueError(f"Ожидался список, получено {type(data)}: {path}")
    return data


# ---------- вспомогательно

# Теги, которые не считаем «словами» — их выбрасываем.
# PUNCT — знаки препинания, SYM — символы, X — прочее (латиница вперемешку и т.п.)
SKIP_POS = {"PUNCT", "SYM", "X"}


def _filter_tokens(tokens: list[dict]) -> list[dict]:
    """Убирает пунктуацию и символы — оставляет только слова."""
    return [t for t in tokens if t["pos"] not in SKIP_POS]


def _split_by_sentences(tokens: list[dict]) -> list[list[dict]]:
    """Группирует токены по предложениям через sent_id."""
    by_sent: dict[int, list[dict]] = defaultdict(list)
    for t in tokens:
        by_sent[t["sent_id"]].append(t)
    return [by_sent[sid] for sid in sorted(by_sent)]


# N-граммы лемм

def metric_word_ngrams(
    tokens: list[dict],
    n: int,
    top_k: int | None = None,
) -> Counter:
    """
    Частоты N-грамм ЛЕММ в одном тексте.

    Параметры:
        tokens — список токенов processed
        n      — длина N-граммы (1, 2, 3)
        top_k  — top_k самых частых

    Возвращает:
        Counter {(lemma1, ..., lemman): count}
    Логика:
        1. Убираем пунктуацию и символы.
        2. Разбиваем на предложения по sent_id.
        3. Внутри каждого предложения собираем N-граммы
           скользящим окном.
    """
    counter: Counter = Counter()
    clean_tokens = _filter_tokens(tokens)

    for sent in _split_by_sentences(clean_tokens):
        lemmas = [t["lemma"] for t in sent]
        for i in range(len(lemmas) - n + 1):
            counter[tuple(lemmas[i:i + n])] += 1

    if top_k is not None:
        return Counter(dict(counter.most_common(top_k)))
    return counter


# ---------- глобальные топы по корпусу

WORD_NGRAM_TOP: dict[int, list[tuple[str, ...]]] = {1: [], 2: [], 3: []}

TOP_K = {
    1: 40,   # 40 самых частых лемм корпуса
    2: 20,    # 20 самых частых биграмм
    3: 10,    # 10 самых частых триграмм
}


def collect_top_word_ngrams(processed_paths: list[Path]) -> dict:
    """
    Первый проход по корпусу: собирает глобальные топ-N N-грамм лемм.

    Возвращает:
        {1: [(lemma,), ...], 2: [(l1, l2), ...], 3: [(l1, l2, l3), ...]}
    """
    top_by_n: dict[int, list[tuple[str, ...]]] = {}

    for n in (1, 2, 3):
        total: Counter = Counter()
        for path in processed_paths:
            try:
                tokens = load_processed(path)
            except ValueError:
                continue
            total.update(metric_word_ngrams(tokens, n, top_k=None))
        top_by_n[n] = [ng for ng, _ in total.most_common(TOP_K[n])]

    return top_by_n


# ---------- признаки ----------

def _ngram_to_colname(n: int, ngram: tuple[str, ...]) -> str:
    """
    Например:

    Для N=1: word1g_и, word1g_в, word1g_не
    Для N=2: word2g_в_общем, word2g_так_сказать
    Для N=3: word3g_в_общем_смысле

    """
    cleaned = [part.replace(" ", "_").replace("-", "_") for part in ngram]
    return f"word{n}g_" + "_".join(cleaned)


def metric_word_ngram_features(tokens: list[dict]) -> dict:
    """
    Доли глобальных топ-k N-грамм лемм в конкретном тексте.

    Для каждой N-граммы из глобального топа считает:
        (частота этой N-граммы в тексте) / (общее число N-грамм в тексте).

    """
    feats: dict = {}

    for n, top_list in WORD_NGRAM_TOP.items():
        local = metric_word_ngrams(tokens, n, top_k=None)
        total = sum(local.values()) or 1

        for ngram in top_list:
            col = _ngram_to_colname(n, ngram)
            feats[col] = local.get(ngram, 0) / total

    return feats


# ---------- фасад

def extract_word_features(processed_path: Path) -> dict:
    """Собирает N-граммы слов из processed-файла."""
    tokens = load_processed(processed_path)
    return metric_word_ngram_features(tokens)