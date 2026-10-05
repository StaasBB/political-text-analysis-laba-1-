from pathlib import Path
from text_processing import process_file


BASE_DIR = Path(__file__).resolve().parent / "political_corpus"

#CLEAN_DIR = BASE_DIR / "texts" / "clean"
CLEAN_DIR = BASE_DIR / "texts" / "clean" / "test"
PROCESSED_DIR = BASE_DIR / "texts" / "processed"


def main():

    if not CLEAN_DIR.exists():
        print(f"ОШИБКА: папка не найдена:\n{CLEAN_DIR}")
        return

    files = list(CLEAN_DIR.rglob("*.txt"))

    if not files:
        print("TXT-файлы не найдены.")
        return

    processed = 0

    for source in files:

        # Сохраняем структуру папок:
        #
        # clean/lenin/Государство и революция.txt
        # ->
        # processed/lenin/Государство и революция.txt

        relative_path = source.relative_to(CLEAN_DIR)
        destination = PROCESSED_DIR / relative_path

        process_file(source, destination)

        processed += 1
        print(f"✓ {relative_path}")

    print()
    print(f"Обработано файлов: {processed}")
    print(f"Результат: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()