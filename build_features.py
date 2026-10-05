"""
Собирает признаки для сегментов корпуса.

Источники признаков:
  1. clean/*.txt
       → 12 существующих стилометрических метрик

  2. processed/*.txt
       → POS-доли
       → POS-счётчики
       → POS N-граммы

  3. clean/*.txt
       → TextRank

  4. processed/*.txt
       → TF-IDF по леммам

ВАЖНО:
  - split берётся из texts_table.csv
  - TextRank vocabulary строится только на TRAIN
  - TF-IDF fit() выполняется только на TRAIN
  - POS N-граммы собираются только из TRAIN
  - TEST используется только для transform/extract
"""


from pathlib import Path
import csv


# ============================================================
# СТИЛОМЕТРИЯ
# ============================================================

from stylometry import (
    extract_features as extract_stylometry_features
)


# ============================================================
# POS-ПРИЗНАКИ
# ============================================================

from stylometry_processed import (
    extract_processed_features,
    collect_top_pos_ngrams,
    POS_NGRAM_TOP,
)


# ============================================================
# WORD N-GRAMS
# ============================================================

from stylometry_words import (
    extract_word_features,
    collect_top_word_ngrams,
    WORD_NGRAM_TOP,
)


# ============================================================
# CONTENT-ПРИЗНАКИ
# ============================================================

from stylometry_content_features import (
    extract_features as extract_content_features,
    collect_textrank_vocabulary,
    create_tfidf_vectorizer,
)


# ============================================================
# ПУТИ
# ============================================================

BASE_DIR = (
    Path(__file__).resolve().parent
    / "political_corpus"
)

CLEAN_DIR = (
    BASE_DIR
    / "texts"
    / "clean"
)

PROCESSED_DIR = (
    BASE_DIR
    / "texts"
    / "processed"
)

TEXTS_TABLE_CSV = (
    BASE_DIR
    #/ "texts_table.csv"
    / "supertest_texts_table.csv"
)

FEATURES_CSV = (
    BASE_DIR
    #/ "features.csv"
    / "supertest_features.csv"
)


# ============================================================
# НАСТРОЙКИ
# ============================================================

TOP_N_GRAMS = 20


# ============================================================
# МЕТАДАННЫЕ
# ============================================================

METADATA_COLUMNS = [
    "segment_id",
    "text_id",
    "segment_number",
    "segment_count",
    "author",
    "filename",
    "line_count",
    "nonempty_line_count",
    "sentence_count",
    "word_count",
    "char_count",
    "quadrant",
    "work",
    "genre",
    "style",
    "split",
]


# ============================================================
# НОРМАЛИЗАЦИЯ ИМЕНИ ФАЙЛА
# ============================================================

def normalize_filename(filename: str) -> str:
    """
    Нормализует имя файла для сопоставления.

    Убирает возможные пробелы по краям
    и приводит разделители к одному виду.
    """

    return (
        str(filename)
        .strip()
        .replace("\\", "/")
        .split("/")[-1]
    )


# ============================================================
# ЗАГРУЗКА TEXTS_TABLE
# ============================================================

def load_texts_table() -> dict:
    """
    Загружает texts_table.csv.

    Сопоставление выполняется по:

        author + filename

    Например:

        atabekian
        +
        Возможна ли анархическая социальная революция_01.txt

    """

    if not TEXTS_TABLE_CSV.exists():
        raise FileNotFoundError(
            f"Не найден файл:\n{TEXTS_TABLE_CSV}"
        )

    metadata = {}

    with TEXTS_TABLE_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(
                "texts_table.csv не содержит заголовок."
            )

        required_columns = {
            "segment_id",
            "author",
            "filename",
            "split",
        }

        missing_columns = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "В texts_table.csv отсутствуют "
                f"колонки: {sorted(missing_columns)}"
            )

        for row in reader:

            author = (
                str(row["author"])
                .strip()
                .lower()
            )

            filename = normalize_filename(
                row["filename"]
            )

            key = (
                author,
                filename
            )

            if key in metadata:
                raise ValueError(
                    "Дубликат в texts_table.csv:\n"
                    f"  author: {author}\n"
                    f"  filename: {filename}"
                )

            split = (
                str(row["split"])
                .strip()
                .lower()
            )

            if split not in {"train", "test"}:
                raise ValueError(
                    f"Неверный split:\n"
                    f"  segment_id: {row['segment_id']}\n"
                    f"  split: {split!r}"
                )

            row["split"] = split

            metadata[key] = row

    return metadata


# ============================================================
# ПОИСК ФАЙЛОВ
# ============================================================

def find_clean_files() -> list[Path]:
    """
    Находит все TXT-файлы в clean.

    Сейчас структура:

        clean/
        ├── author1/
        │   ├── work_01.txt
        │   └── work_02.txt
        │
        └── author2/
            └── work_01.txt

    Никаких segmented/combined.
    """

    if not CLEAN_DIR.exists():
        return []

    return sorted(
        CLEAN_DIR.rglob("*.txt")
    )


# ============================================================
# СОПОСТАВЛЕНИЕ CLEAN С TEXTS_TABLE
# ============================================================

def build_file_metadata(
    clean_files: list[Path],
    metadata: dict
):
    """
    Сопоставляет clean-файлы с texts_table.csv.

    Ключ:

        author + filename
    """

    records = []
    missing_metadata = []

    for clean_path in clean_files:

        relative_path = (
            clean_path.relative_to(
                CLEAN_DIR
            )
        )

        # Автор = первая папка внутри clean
        if len(relative_path.parts) < 2:

            print(
                f"  ⚠ Неверная структура: "
                f"{relative_path}"
            )

            continue

        author = (
            relative_path.parts[0]
            .strip()
            .lower()
        )

        filename = normalize_filename(
            clean_path.name
        )

        key = (
            author,
            filename
        )

        if key not in metadata:

            missing_metadata.append(
                f"{author}/{filename}"
            )

            continue

        records.append(
            (
                clean_path,
                metadata[key]
            )
        )

    return records, missing_metadata


# ============================================================
# TRAIN / TEST
# ============================================================

def split_records(records):

    train = []
    test = []

    for clean_path, row in records:

        if row["split"] == "train":

            train.append(
                (clean_path, row)
            )

        elif row["split"] == "test":

            test.append(
                (clean_path, row)
            )

    return train, test


# ============================================================
# СТАТИСТИКА
# ============================================================

def print_split_statistics(
    train_records,
    test_records
):

    train_words = sum(
        int(row.get("word_count", 0) or 0)
        for _, row in train_records
    )

    test_words = sum(
        int(row.get("word_count", 0) or 0)
        for _, row in test_records
    )

    train_sentences = sum(
        int(row.get("sentence_count", 0) or 0)
        for _, row in train_records
    )

    test_sentences = sum(
        int(row.get("sentence_count", 0) or 0)
        for _, row in test_records
    )

    print()
    print("=" * 70)
    print("TRAIN / TEST")
    print("=" * 70)

    print(
        f"TRAIN: {len(train_records):>3} сегментов, "
        f"{train_words:,} слов, "
        f"{train_sentences:,} предложений"
    )

    print(
        f"TEST:  {len(test_records):>3} сегментов, "
        f"{test_words:,} слов, "
        f"{test_sentences:,} предложений"
    )


# ============================================================
# TRAIN TEXTS
# ============================================================

def collect_train_texts(train_records):

    texts = []

    for clean_path, _ in train_records:

        texts.append(
            clean_path.read_text(
                encoding="utf-8"
            )
        )

    return texts


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    if not CLEAN_DIR.exists():

        print(
            f"Нет папки:\n{CLEAN_DIR}"
        )

        return

    clean_files = find_clean_files()

    if not clean_files:

        print(
            "Нет файлов в texts/clean"
        )

        return

    print(
        f"Найдено файлов в clean: "
        f"{len(clean_files)}"
    )

    # --------------------------------------------------------
    # TEXTS TABLE
    # --------------------------------------------------------

    metadata = load_texts_table()

    print(
        f"Записей в texts_table.csv: "
        f"{len(metadata)}"
    )

    # --------------------------------------------------------
    # СОПОСТАВЛЕНИЕ
    # --------------------------------------------------------

    records, missing_metadata = (
        build_file_metadata(
            clean_files,
            metadata
        )
    )

    if missing_metadata:

        print()
        print(
            "⚠ Для следующих файлов "
            "нет записи в texts_table.csv:"
        )

        for item in missing_metadata:

            print(
                f"  - {item}"
            )

        print()

        print(
            f"Сопоставлено: "
            f"{len(records)}"
        )

        print(
            f"Не сопоставлено: "
            f"{len(missing_metadata)}"
        )

        print()

        print(
            "Эти файлы не будут включены "
            "в features.csv."
        )

    if not records:

        print(
            "ОШИБКА: не удалось сопоставить "
            "ни одного файла."
        )

        return

    # --------------------------------------------------------
    # TRAIN / TEST
    # --------------------------------------------------------

    train_records, test_records = (
        split_records(records)
    )

    if not train_records:

        print(
            "ОШИБКА: TRAIN пуст."
        )

        return

    print_split_statistics(
        train_records,
        test_records
    )

    # ========================================================
    # TRAIN TEXTS
    # ========================================================

    print()
    print("=" * 70)
    print("ПОДГОТОВКА ПРИЗНАКОВ ПО TRAIN")
    print("=" * 70)

    train_texts = collect_train_texts(
        train_records
    )

    print(
        f"TRAIN-текстов загружено: "
        f"{len(train_texts)}"
    )

    # ========================================================
    # TRAIN PROCESSED PATHS
    # ========================================================

    train_processed_paths = []

    for clean_path, _ in train_records:

        # processed имеет ту же структуру,
        # что и clean

        rel = clean_path.relative_to(
            CLEAN_DIR
        )

        proc_path = (
            PROCESSED_DIR
            / rel
        )

        if proc_path.exists():

            train_processed_paths.append(
                proc_path
            )

    print(
        f"TRAIN processed-файлов: "
        f"{len(train_processed_paths)}"
    )

    # ========================================================
    # TEXT RANK
    # ========================================================

    print()
    print(
        "Собираем TextRank vocabulary "
        "по TRAIN..."
    )

    textrank_vocabulary = (
        collect_textrank_vocabulary(
            train_texts
        )
    )

    print(
        f"  TextRank vocabulary: "
        f"{len(textrank_vocabulary)} слов"
    )

    # ========================================================
    # TF-IDF
    # ========================================================

    print()
    print(
        "Обучаем TF-IDF по леммам TRAIN..."
    )

    if not train_processed_paths:

        print(
            "  ⚠ TRAIN processed-файлов нет. "
            "TF-IDF не будет рассчитан."
        )

        tfidf_vectorizer = None
        tfidf_feature_count = 0

    else:

        tfidf_vectorizer = (
            create_tfidf_vectorizer(
                train_processed_paths
            )
        )

        tfidf_feature_count = len(
            tfidf_vectorizer
            .get_feature_names_out()
        )

        print(
            f"  TF-IDF vocabulary: "
            f"{tfidf_feature_count} лемм"
        )

    # ========================================================
    # POS N-GRAMS
    # ========================================================

    print()
    print(
        f"Собираем топ-{TOP_N_GRAMS} "
        "POS N-грамм по TRAIN..."
    )

    if train_processed_paths:

        top = collect_top_pos_ngrams(
            train_processed_paths,
            top_k=TOP_N_GRAMS
        )

        POS_NGRAM_TOP.clear()
        POS_NGRAM_TOP.update(top)

        print(
            f"  Униграмм:  {len(top[1])}"
        )

        print(
            f"  Биграмм:   {len(top[2])}"
        )

        print(
            f"  Триграмм:  {len(top[3])}"
        )

        print(
            f"  4-грамм:   {len(top[4])}"
        )

    else:

        print(
            "  ⚠ TRAIN processed-файлов нет."
        )

    # ========================================================
    # WORD N-GRAMS
    # ========================================================

    print()
    print(
        f"Собираем топ-{TOP_N_GRAMS} "
        "N-грамм слов по TRAIN..."
    )

    if train_processed_paths:

        word_top = collect_top_word_ngrams(
            train_processed_paths
        )

        WORD_NGRAM_TOP.clear()
        WORD_NGRAM_TOP.update(word_top)

        print(
            f"  Униграмм:  {len(word_top[1])}"
        )

        print(
            f"  Биграмм:   {len(word_top[2])}"
        )

        print(
            f"  Триграмм:  {len(word_top[3])}"
        )

    else:

        print(
            "  ⚠ TRAIN processed-файлов нет."
        )

    # ========================================================
    # СБОРКА ПРИЗНАКОВ
    # ========================================================

    print()
    print("=" * 70)
    print("СБОРКА ВЕКТОРОВ")
    print("=" * 70)

    rows = []

    for clean_path, metadata_row in records:

        rel = clean_path.relative_to(
            CLEAN_DIR
        )

        proc_path = (
            PROCESSED_DIR
            / rel
        )

        print()

        print(
            f"[{metadata_row['split'].upper():5}] "
            f"{metadata_row['segment_id']}"
        )

        # ----------------------------------------------------
        # STYLOMETRY
        # ----------------------------------------------------

        feats = (
            extract_stylometry_features(
                clean_path
            )
        )

        # ----------------------------------------------------
        # POS
        # ----------------------------------------------------

        if proc_path.exists():

            try:

                feats.update(
                    extract_processed_features(
                        proc_path
                    )
                )

            except Exception as e:

                print(
                    f"  ⚠ POS не прочитан: "
                    f"{e}"
                )

        else:

            print(
                f"  ⚠ нет processed: "
                f"{rel}"
            )

        # ----------------------------------------------------
        # WORD N-GRAMS
        # ----------------------------------------------------

        if proc_path.exists():

            try:

                feats.update(
                    extract_word_features(
                        proc_path
                    )
                )

            except Exception as e:

                print(
                    f"  ⚠ N-граммы слов не рассчитаны: "
                    f"{e}"
                )

        # ----------------------------------------------------
        # TEXT RANK + TF-IDF
        # ----------------------------------------------------

        try:

            if tfidf_vectorizer is not None and proc_path.exists():

                feats.update(
                    extract_content_features(
                        clean_path,
                        proc_path,
                        textrank_vocabulary,
                        tfidf_vectorizer
                    )
                )

            else:

                # TextRank можно рассчитать независимо
                # от наличия processed-файла / TF-IDF.
                from stylometry_content_features import (
                    metric_textrank
                )

                feats.update(
                    metric_textrank(
                        clean_path.read_text(
                            encoding="utf-8"
                        ),
                        textrank_vocabulary
                    )
                )

        except Exception as e:

            print(
                f"  ⚠ content features "
                f"не рассчитаны: {e}"
            )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        row = {}

        for column in METADATA_COLUMNS:

            row[column] = (
                metadata_row.get(
                    column,
                    ""
                )
            )

        row["file"] = str(rel)

        # Признаки
        row.update(feats)

        rows.append(row)

        print(
            "  ✓ готово"
        )

    # ========================================================
    # ПРОВЕРКА
    # ========================================================

    if not rows:

        print(
            "ОШИБКА: строки не созданы."
        )

        return

    # ========================================================
    # ОБЩИЙ СПИСОК ПРИЗНАКОВ
    # ========================================================

    feature_names = set()

    for row in rows:

        for key in row:

            if key not in METADATA_COLUMNS:
                if key != "file":

                    feature_names.add(
                        key
                    )

    feature_names = sorted(
        feature_names
    )

    # ========================================================
    # ЗАПОЛНЯЕМ ПРОПУСКИ НУЛЯМИ
    # ========================================================

    for row in rows:

        for feature_name in feature_names:

            if feature_name not in row:

                row[feature_name] = 0.0

    # ========================================================
    # КОЛОНКИ CSV
    # ========================================================

    fieldnames = (
        METADATA_COLUMNS
        + ["file"]
        + feature_names
    )

    # ========================================================
    # СОХРАНЕНИЕ
    # ========================================================

    with FEATURES_CSV.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # ИТОГ
    # ========================================================

    train_count = sum(
        1
        for row in rows
        if row["split"] == "train"
    )

    test_count = sum(
        1
        for row in rows
        if row["split"] == "test"
    )

    print()
    print("=" * 70)
    print("ГОТОВО")
    print("=" * 70)

    print(
        f"Файл:              {FEATURES_CSV}"
    )

    print(
        f"Сегментов:         {len(rows)}"
    )

    print(
        f"TRAIN:             {train_count}"
    )

    print(
        f"TEST:              {test_count}"
    )

    print(
        f"Признаков:         {len(feature_names)}"
    )

    print()
    print(
        "Обучение пространства признаков:"
    )

    print(
        "  ✓ TextRank vocabulary → TRAIN"
    )

    print(
        "  ✓ TF-IDF fit по леммам → TRAIN"
    )

    print(
        "  ✓ POS N-граммы → TRAIN"
    )

    print(
        "  ✓ N-граммы слов → TRAIN"
    )

    print()
    print(
        "TEST не использовался "
        "при построении словарей."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()