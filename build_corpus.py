from pathlib import Path
import re


# Корень проекта
BASE_DIR = Path(__file__).resolve().parent / "political_corpus"

RAW_DIR = BASE_DIR / "texts" / "raw"
CLEAN_DIR = BASE_DIR / "texts" / "clean"


def remove_square_and_curly(text: str) -> str:
    """Удаляет содержимое [] и {} вместе со скобками."""

    # Удаляем содержимое квадратных скобок
    text = re.sub(r"\[[^\]]*\]", "", text)

    # Удаляем содержимое фигурных скобок
    text = re.sub(r"\{[^}]*\}", "", text)

    return text


def remove_page_references(text: str) -> str:
    """
    Удаляет круглые скобки, если внутри них встречается
    'стр' или 'стр.'.

    Например:
    (стр. 25) -> удаляется
    (см. стр. 25) -> удаляется
    (с 1902 года) -> остается
    (т. е. государство) -> остается
    """

    pattern = r"\([^()]*\bстр\.?\b[^()]*\)"

    return re.sub(
        pattern,
        "",
        text,
        flags=re.IGNORECASE
    )


def clean_line(line: str) -> str:
    """Очистка одного абзаца."""

    # Убираем BOM
    line = line.replace("\ufeff", "")

    # [] и {} удаляются полностью
    line = remove_square_and_curly(line)

    # () удаляются только если внутри есть 'стр' / 'стр.'
    line = remove_page_references(line)

    # Табуляции -> пробел
    line = line.replace("\t", " ")

    # Несколько пробелов -> один
    line = re.sub(r" {2,}", " ", line)

    # Убираем пробел перед знаками препинания
    line = re.sub(r"\s+([,.!?;:])", r"\1", line)

    # Убираем пробелы по краям
    line = line.strip()

    return line


def is_separator(line: str) -> bool:
    """Удаляет технические разделители вроде * * *."""

    return bool(re.fullmatch(r"\s*(?:\*\s*){3,}", line))


def process_file(source: Path, destination: Path):
    """Обрабатывает один txt-файл."""

    try:
        text = source.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        # Запасной вариант для старых русских файлов
        text = source.read_text(encoding="cp1251")

    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:

        # Удаляем технические разделители
        if is_separator(line):
            continue

        line = clean_line(line)

        # Пустые строки не сохраняем
        if not line:
            continue

        cleaned_lines.append(line)

    # Каждая строка = отдельный абзац
    result = "\n".join(cleaned_lines)

    destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(
        result + "\n",
        encoding="utf-8"
    )


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
        destination = CLEAN_DIR / relative_path

        process_file(source, destination)

        processed += 1
        print(f"✓ {relative_path}")

    print()
    print(f"Обработано файлов: {processed}")
    print(f"Результат: {CLEAN_DIR}")


if __name__ == "__main__":
    main()