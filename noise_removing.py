from pathlib import Path
import re

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


def remove_noise_from_file(source: Path, destination: Path):
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
