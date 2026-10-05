from pathlib import Path
import re
import ast

from summa.keywords import keywords
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# НАСТРОЙКИ
# ============================================================

# Сколько ключевых слов TextRank брать из одного текста
TEXTRANK_TOP_N = 50

# Сколько TextRank-слов оставить в общем словаре признаков
TEXTRANK_VOCAB_SIZE = 100

# Максимальное количество TF-IDF-признаков
TFIDF_MAX_FEATURES = 150


# ============================================================
# ЗАГРУЗКА CLEAN ТЕКСТА
# ============================================================

def load_clean(clean_path: Path) -> str:
    """
    Загружает очищенный текст сегмента.
    """

    return clean_path.read_text(
        encoding="utf-8"
    )


# ============================================================
# ЗАГРУЗКА ЛЕММ ИЗ PROCESSED
# ============================================================

def load_processed_lemmas(processed_path: Path) -> str:
    """
    Загружает processed-файл и возвращает
    строку из лемм.

    Processed-файл имеет вид:

        [
            {
                'origin': 'Что',
                'lemma': 'что',
                'pos': 'PRON',
                'sent_id': 0
            },
            ...
        ]

    Для TF-IDF используется только поле 'lemma'.
    """

    text = processed_path.read_text(
        encoding="utf-8"
    )

    try:
        words = ast.literal_eval(text)

    except (SyntaxError, ValueError) as e:
        raise ValueError(
            f"Не удалось прочитать processed-файл:\n"
            f"{processed_path}"
        ) from e

    lemmas = []

    for word in words:

        lemma = str(
            word.get("lemma", "")
        ).lower().strip()

        if not lemma:
            continue

        lemma = re.sub(
            r"[^а-яёa-z0-9-]",
            "",
            lemma
        )

        if lemma:
            lemmas.append(lemma)

    return " ".join(lemmas)


# ============================================================
# НОРМАЛИЗАЦИЯ СЛОВА
# ============================================================

def normalize_word(word: str) -> str:
    """
    Приводит слово к единому виду.

    Например:
        "Государство" -> "государство"
        "власть,"     -> "власть"
    """

    word = str(word).lower().strip()

    word = re.sub(
        r"[^а-яёa-z0-9-]",
        "",
        word
    )

    return word


# ============================================================
# ПОЛУЧЕНИЕ TEXT RANK ДЛЯ ОДНОГО ТЕКСТА
# ============================================================

def get_textrank_scores(
    text: str,
    top_n: int = TEXTRANK_TOP_N
) -> dict:
    """
    Возвращает словарь {слово: score} для одного текста.
    """
    features = {}

    if not text.strip():
        return features

    try:
        result = keywords(
            text,
            words=top_n,
            language="russian",   # ← обязательно для русского
            split=True,
            scores=True,
        )
    except Exception as e:
        print(f"  ⚠ TextRank FAIL: {type(e).__name__}: {e}")
        return features

    if not result:
        print(f"  ⚠ TextRank пуст (длина текста: {len(text.split())} слов)")
        return features

    for word, score in result:
        word = normalize_word(word)
        if not word:
            continue
        features[word] = float(score)

    return features

# ============================================================
# СОЗДАНИЕ ОБЩЕГО TEXT RANK VOCABULARY
# ============================================================

def collect_textrank_vocabulary(
    texts: list[str],
    top_k: int = TEXTRANK_VOCAB_SIZE
) -> list[str]:
    """
    Создаёт общий словарь TextRank-признаков.

    ВАЖНО:
    передаваемые texts должны быть только TRAIN.

    Каждый train-текст отдельно обрабатывается TextRank,
    после чего его ключевые слова объединяются.

    Для каждого слова суммируется TextRank-оценка
    по всем текстам.

    Затем выбираются top_k наиболее характерных слов.

    Возвращает:

        [
            "государство",
            "власть",
            "общество",
            ...
        ]
    """

    total_scores = {}

    for text in texts:

        scores = get_textrank_scores(text)

        for word, score in scores.items():

            total_scores[word] = (
                total_scores.get(word, 0.0)
                + score
            )

    sorted_words = sorted(
        total_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    vocabulary = [
        word
        for word, _ in sorted_words[:top_k]
    ]

    return vocabulary


# ============================================================
# TEXT RANK ПРИЗНАКИ ОДНОГО ТЕКСТА
# ============================================================

def metric_textrank(
    text: str,
    vocabulary: list[str]
) -> dict:
    """
    Создаёт фиксированный TextRank-вектор одного текста.

    Все слова из vocabulary присутствуют в результате.

    Если слово отсутствует среди ключевых слов текста:
        значение = 0.0

    Например:

        {
            "textrank_государство": 0.083,
            "textrank_власть": 0.051,
            "textrank_революция": 0.0
        }
    """

    scores = get_textrank_scores(text)

    features = {}

    for word in vocabulary:

        score = scores.get(
            word,
            0.0
        )

        features[
            f"textrank_{word}"
        ] = float(score)

    return features


# ============================================================
# СОЗДАНИЕ TF-IDF VECTORIZER
# ============================================================

def create_tfidf_vectorizer(
    processed_paths: list[Path]
) -> TfidfVectorizer:
    """
    Создаёт и обучает TF-IDF vectorizer
    на леммах из processed-файлов.

    ВАЖНО:
    processed_paths должны содержать ТОЛЬКО TRAIN.

    После fit() этот vectorizer используется
    для train/test/новых текстов через transform().

    Каждая форма слова после лемматизации
    считается одной леммой.

    Например:

        государство
        государства
        государству
        государством

    превращаются в:

        государство
    """

    if not processed_paths:
        raise ValueError(
            "Невозможно обучить TF-IDF: "
            "список TRAIN processed-файлов пуст."
        )

    texts = [
        load_processed_lemmas(path)
        for path in processed_paths
    ]

    vectorizer = TfidfVectorizer(
        lowercase=False,

        token_pattern=r"(?u)\b[а-яёa-z0-9-]+\b",

        max_features=TFIDF_MAX_FEATURES,

        ngram_range=(1, 1)
    )

    vectorizer.fit(texts)

    return vectorizer


# ============================================================
# TF-IDF ПРИЗНАКИ ОДНОГО ТЕКСТА
# ============================================================

def metric_tfidf(
    processed_path: Path,
    vectorizer: TfidfVectorizer
) -> dict:
    """
    Получает TF-IDF-вектор одного текста
    по его леммам.

    Используется уже обученный vectorizer,
    поэтому здесь выполняется только transform().
    """

    text = load_processed_lemmas(
        processed_path
    )

    vector = vectorizer.transform(
        [text]
    )

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    values = vector.toarray()[0]

    features = {}

    for lemma, value in zip(
        feature_names,
        values
    ):

        features[
            f"tfidf_{lemma}"
        ] = float(value)

    return features


# ============================================================
# ПОЛНЫЕ CONTENT FEATURES
# ============================================================

def extract_features(
    clean_path: Path,
    processed_path: Path,
    textrank_vocabulary: list[str],
    tfidf_vectorizer: TfidfVectorizer
) -> dict:
    """
    Собирает все content-признаки одного сегмента:

        1. TextRank
        2. TF-IDF по леммам

    TextRank работает с clean-текстом.

    TF-IDF работает с леммами из processed-текста.

    Возвращает единый словарь признаков.
    """

    clean_text = load_clean(
        clean_path
    )

    features = {}

    # --------------------------------------------------------
    # TEXT RANK
    # --------------------------------------------------------

    features.update(
        metric_textrank(
            clean_text,
            textrank_vocabulary
        )
    )

    # --------------------------------------------------------
    # TF-IDF ПО ЛЕММАМ
    # --------------------------------------------------------

    features.update(
        metric_tfidf(
            processed_path,
            tfidf_vectorizer
        )
    )

    return features