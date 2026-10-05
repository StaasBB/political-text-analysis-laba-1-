from pathlib import Path
import re
import pandas as pd

#  неработает удалил старые таблицы
# ============================================================
# ПУТИ
# ============================================================

BASE_DIR = Path(__file__).resolve().parent / "political_corpus"

# Старая таблица с метаданными произведений
OLD_STATS_FILE = BASE_DIR / "text_stats_train_test_split.csv"

# Очищенные и сегментированные тексты
CLEAN_DIR = BASE_DIR / "texts" / "clean"

# Новая гигатаблица
OUTPUT_FILE = BASE_DIR / "texts_table.csv"


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def count_words(text: str) -> int:
    """Количество слов."""

    return len(
        re.findall(
            r"\b\w+\b",
            text,
            flags=re.UNICODE
        )
    )


def count_sentences(text: str) -> int:
    """Примерный подсчёт предложений."""

    sentences = re.findall(
        r"[^.!?]+[.!?]+",
        text,
        flags=re.UNICODE
    )

    return len(sentences)


def get_original_filename(filename: str) -> str:
    """
    Убирает номер сегмента из имени файла.

    Например:

        Азбука коммунизма_01.txt
        Азбука коммунизма_02.txt
        Азбука коммунизма_15.txt

    превращаются в:

        Азбука коммунизма.txt
    """

    path = Path(filename)

    stem = path.stem

    # Убираем последний _01, _02, _03...
    stem = re.sub(
        r"_\d+$",
        "",
        stem
    )

    return stem + path.suffix


def normalize_name(name: str) -> str:
    """
    Нормализация имени файла для сопоставления.

    Нужна на случай небольших различий:
    регистр, лишние пробелы и т. п.
    """

    name = str(name)

    name = Path(name).stem

    name = name.lower()

    name = re.sub(
        r"\s+",
        " ",
        name
    )

    name = name.strip()

    return name


# ============================================================
# ЗАГРУЗКА СТАРОЙ ТАБЛИЦЫ
# ============================================================

def load_old_stats():

    if not OLD_STATS_FILE.exists():

        raise FileNotFoundError(
            f"Не найдена старая таблица:\n"
            f"{OLD_STATS_FILE}"
        )

    df = pd.read_csv(
        OLD_STATS_FILE,
        encoding="utf-8-sig"
    )

    required_columns = [
        "text_id",
        "author",
        "filename",
        "quadrant",
        "work",
        "genre",
        "style",
        "split"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "В text_stats_train_test_split.csv "
            "отсутствуют столбцы:\n"
            + ", ".join(missing)
        )

    return df


# ============================================================
# ПОСТРОЕНИЕ ГИГАТАБЛИЦЫ
# ============================================================

def build_table(old_stats):

    rows = []

    files = list(
        CLEAN_DIR.rglob("*.txt")
    )

    if not files:

        raise FileNotFoundError(
            f"В папке clean нет TXT-файлов:\n"
            f"{CLEAN_DIR}"
        )

    print(
        f"Найдено сегментов: {len(files)}"
    )

    # --------------------------------------------------------
    # Сколько сегментов у каждого произведения
    # --------------------------------------------------------

    segment_counts = {}

    for file in files:

        original_filename = get_original_filename(
            file.name
        )

        author = file.parent.name

        key = (
            author.strip().lower(),
            normalize_name(original_filename)
        )

        segment_counts[key] = (
            segment_counts.get(key, 0) + 1
        )

    # --------------------------------------------------------
    # Создаём справочник:
    #
    # author + название произведения
    #                 ↓
    #             metadata
    # --------------------------------------------------------

    metadata = {}

    for _, row in old_stats.iterrows():

        key = (
            str(row["author"]).strip().lower(),
            normalize_name(row["filename"])
        )

        metadata[key] = row

    # --------------------------------------------------------
    # Обрабатываем сегменты
    # --------------------------------------------------------

    for file in files:

        filename = file.name

        # Автор берётся из структуры папок:
        #
        # clean/bukharin/Азбука коммунизма_01.txt
        #
        #                         ↓
        #
        # author = bukharin

        # --------------------------------------------------------
        # Номер сегмента
        # --------------------------------------------------------

        segment_number_match = re.search(
            r"_(\d+)\.txt$",
            filename
        )

        if segment_number_match:
            segment_number = int(
                segment_number_match.group(1)
            )
        else:
            segment_number = 1

        author = file.parent.name

        original_filename = get_original_filename(
            filename
        )

        key = (
            author.strip().lower(),
            normalize_name(original_filename)
        )
        segment_count = segment_counts[key]
        # ----------------------------------------------------
        # Ищем произведение в старой таблице
        # ----------------------------------------------------

        if key not in metadata:

            print(
                f"⚠ НЕ НАЙДЕНО СОПОСТАВЛЕНИЕ:"
            )

            print(
                f"   Автор: {author}"
            )

            print(
                f"   Файл:  {filename}"
            )

            print(
                f"   Ищем:  {original_filename}"
            )

            continue

        old_row = metadata[key]
        segment_id = (f"{old_row['text_id']}_{segment_number:02d}")

        # ----------------------------------------------------
        # Читаем сегмент
        # ----------------------------------------------------

        try:

            text = file.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:

            text = file.read_text(
                encoding="cp1251"
            )

        lines = text.splitlines()

        # ----------------------------------------------------
        # Статистика сегмента
        # ----------------------------------------------------

        line_count = len(lines)

        nonempty_line_count = sum(
            bool(line.strip())
            for line in lines
        )

        sentence_count = count_sentences(
            text
        )

        word_count = count_words(
            text
        )

        char_count = len(text)

        # ----------------------------------------------------
        # Добавляем строку
        # ----------------------------------------------------

        rows.append({
            "segment_id":
                segment_id,

            "text_id":
                old_row["text_id"],

            "segment_number":
                segment_number,

            "segment_count":
                segment_count,

            # Автор
            "author":
                old_row["author"],

            # Конкретный сегмент
            "filename":
                filename,

            # Статистика сегмента
            "line_count":
                line_count,

            "nonempty_line_count":
                nonempty_line_count,

            "sentence_count":
                sentence_count,

            "word_count":
                word_count,

            "char_count":
                char_count,

            # Метаданные исходного произведения
            "quadrant":
                old_row["quadrant"],

            "work":
                old_row["work"],

            "genre":
                old_row["genre"],

            "style":
                old_row["style"],

            # Split исходного произведения
            "split":
                old_row["split"]
        })

    return pd.DataFrame(rows)


# ============================================================
# ВЫВОД СТАТИСТИКИ
# ============================================================

def print_summary(df):

    print()
    print("=" * 80)
    print("ИТОГОВАЯ ГИГАТАБЛИЦА")
    print("=" * 80)

    print(
        f"Сегментов:             {len(df)}"
    )

    print(
        f"Произведений:           "
        f"{df['text_id'].nunique()}"
    )

    print(
        f"Авторов:                "
        f"{df['author'].nunique()}"
    )

    print(
        f"Всего слов:             "
        f"{df['word_count'].sum():,}"
    )

    # --------------------------------------------------------
    # TRAIN / TEST
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("TRAIN / TEST")
    print("-" * 80)

    for split in ["TRAIN", "TEST"]:

        part = df[
            df["split"]
            .astype(str)
            .str.upper()
            == split
        ]

        print(
            f"{split}: "
            f"{len(part)} сегментов / "
            f"{part['text_id'].nunique()} произведений / "
            f"{part['word_count'].sum():,} слов"
        )

    # --------------------------------------------------------
    # ПО КВАДРАНТАМ
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("ПО КВАДРАНТАМ")
    print("-" * 80)

    quadrant_table = (
        df.groupby(
            ["quadrant", "split"]
        )
        .agg(
            segments=("filename", "count"),
            works=("text_id", "nunique"),
            words=("word_count", "sum")
        )
        .reset_index()
    )

    print(
        quadrant_table.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # ПО АВТОРАМ
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("ПО АВТОРАМ")
    print("-" * 80)

    author_table = (
        df.groupby(
            ["author", "split"]
        )
        .agg(
            segments=("filename", "count"),
            works=("text_id", "nunique"),
            words=("word_count", "sum")
        )
        .reset_index()
    )

    print(
        author_table.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("СОЗДАНИЕ TEXTS_TABLE.CSV")
    print("=" * 80)

    # --------------------------------------------------------
    # Загружаем старую таблицу
    # --------------------------------------------------------

    old_stats = load_old_stats()

    print(
        f"Загружено произведений из "
        f"text_stats_train_test_split.csv: "
        f"{len(old_stats)}"
    )

    # --------------------------------------------------------
    # Строим новую таблицу
    # --------------------------------------------------------

    df = build_table(
        old_stats
    )

    if df.empty:

        print()
        print(
            "ОШИБКА: не удалось создать ни одной строки."
        )

        return

    # --------------------------------------------------------
    # Сортировка
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "quadrant",
            "author",
            "text_id",
            "filename"
        ]
    )

    # --------------------------------------------------------
    # Сохранение
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        f"✓ Таблица сохранена:\n"
        f"{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Итог
    # --------------------------------------------------------

    print_summary(df)


if __name__ == "__main__":
    main()
