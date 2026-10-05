"""
Метрики из processed-файлов.

Реализовано:
  1. POS-доли по 5 ключевым частям речи.
  2. POS-счётчики по тем же 5 частям речи.
  3. POS N-граммы (1, 2, 3, 4) — частоты последовательностей частей речи.

Из POS оставлены только: NOUN, VERB, ADJ, ADV, PRON.
"""

from pathlib import Path
import ast
from collections import Counter, defaultdict


KEY_POS = ["NOUN", "VERB", "ADJ", "ADV", "PRON"]


def load_processed(path: Path) -> list[dict]:
    """Читает processed-файл"""
    text = path.read_text(encoding="utf-8")
    data = ast.literal_eval(text)
    if not isinstance(data, list):
        raise ValueError(f"Ожидался список, получено {type(data)}: {path}")
    return data


# --------POS-доли

def metric_pos_distribution(tokens: list[dict]) -> dict:
    """Доля каждой из 5 ключевых частей речи среди всех токенов."""
    counter = Counter(t["pos"] for t in tokens)
    total = sum(counter.values()) or 1
    return {f"pos_{tag}": counter.get(tag, 0) / total for tag in KEY_POS}


# ----------POS-счётчики

def metric_pos_counts(tokens: list[dict]) -> dict:
    """Абсолютное количество токенов каждой из 5 ключевых частей речи."""
    counter = Counter(t["pos"] for t in tokens)
    return {f"n_{tag}": counter.get(tag, 0) for tag in KEY_POS}


# ----------POS N-граммы

def _split_by_sentences(tokens: list[dict]) -> list[list[dict]]:
    """Группирует токены через sent_id."""
    by_sent: dict[int, list[dict]] = defaultdict(list)
    for t in tokens:
        by_sent[t["sent_id"]].append(t)
    return [by_sent[sid] for sid in sorted(by_sent)]


def metric_pos_ngrams(tokens: list[dict], n: int, top_k: int | None = None) -> Counter:
    """
    Частоты POS N-грамм длины n.
    В N-граммах участвуют все теги (включая PUNCT, ADP и т.д.) —
    """
    counter: Counter = Counter()
    for sent in _split_by_sentences(tokens):
        pos_seq = [t["pos"] for t in sent]
        for i in range(len(pos_seq) - n + 1):
            counter[tuple(pos_seq[i:i + n])] += 1
    if top_k is not None:
        return Counter(dict(counter.most_common(top_k)))
    return counter


# ---------- глобальный топ N-грамм по корпусу

POS_NGRAM_TOP: dict[int, list[tuple[str, ...]]] = {1: [], 2: [], 3: [], 4: []}


def collect_top_pos_ngrams(processed_paths: list[Path], top_k: int = 20) -> dict:
    """Первый проход: собирает глобальные топ-K N-грамм каждой длины."""
    top_by_n: dict[int, list[tuple[str, ...]]] = {}
    for n in (1, 2, 3, 4):
        total: Counter = Counter()
        for path in processed_paths:
            try:
                tokens = load_processed(path)
            except ValueError:
                continue
            total.update(metric_pos_ngrams(tokens, n, top_k=None))
        top_by_n[n] = [ng for ng, _ in total.most_common(top_k)]
    return top_by_n


def _ngram_to_colname(n: int, ngram: tuple[str, ...]) -> str:
    """Имя колонки"""
    return f"pos{n}g_" + "_".join(ngram)


def metric_pos_ngram_features(tokens: list[dict]) -> dict:
    """Доли глобальных топ-N N-грамм в конкретном тексте."""
    feats: dict = {}
    for n, top_list in POS_NGRAM_TOP.items():
        local = metric_pos_ngrams(tokens, n, top_k=None)
        total = sum(local.values()) or 1
        for ngram in top_list:
            feats[_ngram_to_colname(n, ngram)] = local.get(ngram, 0) / total
    return feats


# ---------- фасад

def extract_processed_features(processed_path: Path) -> dict:
    """Собирает все метрики из processed-файла в один словарь."""
    tokens = load_processed(processed_path)
    feats = {}
    feats.update(metric_pos_distribution(tokens))   # 5 колонок
    feats.update(metric_pos_counts(tokens))         # 5 колонок
    feats.update(metric_pos_ngram_features(tokens)) # 20×4 = 80 колонок
    return feats