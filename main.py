from pathlib import Path
from noise_removing import remove_noise_from_file
from text_processing import process_file

BASE_DIR = Path(__file__).resolve().parent / "political_corpus"

RAW_DIR = BASE_DIR / "texts" / "raw"
CLEAN_DIR = BASE_DIR / "texts" / "clean"
PROCESSED_DIR = BASE_DIR / "texts" / "processed"


def main():

    if not RAW_DIR.exists():
        print(f"ОШИБКА: папка не найдена:\n{RAW_DIR}")
        return

    files = list(RAW_DIR.rglob("*.txt"))

    if not files:
        print("TXT-файлы не найдены.")
        return

    processed = 0

    for source in files:

        # Сохраняем структуру папок:
        #
        # raw/lenin/Государство и революция.txt
        # ->
        # clean/lenin/Государство и революция.txt

        relative_path = source.relative_to(RAW_DIR)
        noise_destination = CLEAN_DIR / relative_path
        process_destination = PROCESSED_DIR / relative_path

        remove_noise_from_file(source, noise_destination)
        process_file(source, process_destination)

        processed += 1
        print(f"✓ {relative_path}")

    print()
    print(f"Обработано файлов: {processed}")
    print(f"Результат: {CLEAN_DIR}")


if __name__ == "__main__":
    main()