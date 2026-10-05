from pathlib import Path
import re
import shutil

from noise_removing import remove_noise_from_file


# ============================================================
# НАСТРОЙКИ
# ============================================================

BASE_DIR = Path(__file__).resolve().parent / "political_corpus"

RAW_DIR = BASE_DIR / "texts" / "raw"
TEMP_DIR = BASE_DIR / "texts" / "_cleaned_temp"
CLEAN_DIR = BASE_DIR / "texts" / "clean"

# Размер обычного сегмента
MIN_SEGMENT_WORDS = 5000

# Если последний сегмент меньше этого размера,
# присоединяем его к предыдущему
MIN_LAST_SEGMENT_WORDS = 2500


# ============================================================
# ПОДСЧЁТ СЛОВ
# ============================================================

def count_words(text: str) -> int:
    """Считает слова в тексте."""

    return len(
        re.findall(
            r"\b\w+\b",
            text,
            flags=re.UNICODE
        )
    )


# ============================================================
# РАЗБИЕНИЕ ТЕКСТА
# ============================================================

def split_into_segments(
    lines: list[str],
    min_words: int = MIN_SEGMENT_WORDS,
    min_last_words: int = MIN_LAST_SEGMENT_WORDS
) -> list[list[str]]:

    segments = []

    current_segment = []
    current_words = 0

    for line in lines:

        words = count_words(line)

        current_segment.append(line)
        current_words += words

        # Достигли 5000 слов
        if current_words >= min_words:

            segments.append(current_segment)

            current_segment = []
            current_words = 0

    # Остался последний кусок
    if current_segment:

        last_words = sum(
            count_words(line)
            for line in current_segment
        )

        # Если он слишком маленький —
        # присоединяем к предыдущему
        if (
            segments
            and last_words < min_last_words
        ):
            segments[-1].extend(current_segment)

        else:
            segments.append(current_segment)

    return segments


# ============================================================
# СЕГМЕНТАЦИЯ ОДНОГО ОЧИЩЕННОГО ФАЙЛА
# ============================================================

def segment_file(
    source: Path,
    destination_dir: Path
) -> int:

    text = source.read_text(
        encoding="utf-8"
    )

    lines = text.splitlines()

    if not lines:
        return 0

    segments = split_into_segments(lines)

    destination_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    base_name = source.stem

    for number, segment in enumerate(
        segments,
        start=1
    ):

        filename = f"{base_name}_{number:02d}.txt"

        destination = destination_dir / filename

        destination.write_text(
            "\n".join(segment) + "\n",
            encoding="utf-8"
        )

        word_count = sum(
            count_words(line)
            for line in segment
        )

        print(
            f"    → {filename}: "
            f"{word_count:,} слов"
        )

    return len(segments)


# ============================================================
# MAIN
# ============================================================

def main():

    if not RAW_DIR.exists():

        print(
            f"ОШИБКА: папка не найдена:\n"
            f"{RAW_DIR}"
        )

        return

    # --------------------------------------------------------
    # Удаляем старые временные и clean-файлы
    # --------------------------------------------------------

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    if CLEAN_DIR.exists():
        shutil.rmtree(CLEAN_DIR)

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Находим raw-файлы
    # --------------------------------------------------------

    files = list(
        RAW_DIR.rglob("*.txt")
    )

    if not files:

        print(
            "TXT-файлы в raw не найдены."
        )

        return

    total_works = 0
    total_segments = 0

    print("=" * 70)
    print("ОЧИСТКА И СЕГМЕНТАЦИЯ КОРПУСА")
    print("=" * 70)

    # ========================================================
    # ЭТАП 1. ОЧИСТКА
    # ========================================================

    print()
    print("ЭТАП 1. ОЧИСТКА")
    print("-" * 70)

    for source in files:

        relative_path = source.relative_to(
            RAW_DIR
        )

        print(
            f"Очистка: {relative_path}"
        )

        # Сохраняем структуру авторов
        relative_parent = (
            source.parent.relative_to(RAW_DIR)
        )

        temp_destination_dir = (
            TEMP_DIR / relative_parent
        )

        temp_destination = (
            temp_destination_dir / source.name
        )

        # Используем ТУ ЖЕ функцию,
        # которая раньше всё нормально чистила
        remove_noise_from_file(
            source,
            temp_destination
        )

    # ========================================================
    # ЭТАП 2. СЕГМЕНТАЦИЯ
    # ========================================================

    print()
    print("ЭТАП 2. СЕГМЕНТАЦИЯ")
    print("-" * 70)

    temp_files = list(
        TEMP_DIR.rglob("*.txt")
    )

    for source in temp_files:

        relative_path = source.relative_to(
            TEMP_DIR
        )

        print()
        print(
            f"Сегментация: {relative_path}"
        )

        relative_parent = (
            source.parent.relative_to(TEMP_DIR)
        )

        destination_dir = (
            CLEAN_DIR / relative_parent
        )

        segments = segment_file(
            source,
            destination_dir
        )

        total_works += 1
        total_segments += segments

    # ========================================================
    # ЭТАП 3. УДАЛЯЕМ ВРЕМЕННУЮ ПАПКУ
    # ========================================================

    print()
    print("Удаление временных файлов...")

    shutil.rmtree(TEMP_DIR)

    # ========================================================
    # ИТОГ
    # ========================================================

    print()
    print("=" * 70)
    print("ГОТОВО")
    print("=" * 70)

    print(
        f"Исходных произведений: {total_works}"
    )

    print(
        f"Получено сегментов:    {total_segments}"
    )

    print(
        f"Размер сегмента:       ≥ {MIN_SEGMENT_WORDS:,} слов"
    )

    print(
        f"Минимум последнего:    {MIN_LAST_SEGMENT_WORDS:,} слов"
    )

    print(
        f"Результат:             {CLEAN_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()