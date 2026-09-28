from pathlib import Path
import re


def remove_square_and_curly(text: str) -> str:
    """Удаляет содержимое [] и {} вместе со скобками."""

    # Квадратные скобки
    text = re.sub(r"\[[^\]]*\]", "", text)

    # Фигурные скобки
    text = re.sub(r"\{[^}]*\}", "", text)

    # Скобки <<>>
    text = re.sub(r"<<[^>]*>>", "", text)

    return text


def remove_page_references(text: str) -> str:
    """
    Удаляет круглые скобки, если внутри них встречается:
    стр / стр.
    см / см.

    Например:
    (стр. 25) -> удаляется
    (см. стр. 25) -> удаляется
    (с 1902 года) -> остается
    (т. е. государство) -> остается
    """

    pattern = r"\([^()]*\b(?:стр\.?|см\.?)\b[^()]*\)"

    return re.sub(pattern, "", text, flags=re.IGNORECASE)


def remove_strict_parentheses(text: str) -> str:
    """
    Более строгая обработка круглых скобок.

    Удаляет ВСЮ скобку, если внутри есть:
    - хотя бы одна цифра
    - двойные кавычки "

    Примеры:

    (123)                  -> удалить
    (стр. 25)              -> удалить
    (1902 год)             -> удалить
    ("цитата")             -> удалить
    (см. "Государство")    -> удалить
    (т. е. 1902 года)      -> удалить

    Но:

    (государство)          -> оставить
    (т. е. государство)    -> оставить
    (буржуазия)            -> оставить
    """

    # Ищем круглые скобки без вложенных круглых скобок.
    # Если внутри есть цифра ИЛИ двойная кавычка — удаляем всю конструкцию.

    pattern = r'\([^()]*[0-9"][^()]*\)'

    return re.sub(pattern, "", text)


def clean_line(line: str, strict_parentheses: bool = False) -> str:
    """Очистка одного абзаца."""

    # Убираем BOM
    line = line.replace("\ufeff", "")

    # [] и {} удаляются полностью
    line = remove_square_and_curly(line)

    # Обычное удаление ссылок внутри ()
    line = remove_page_references(line)

    # Для специального файла — ужесточаем правила ()
    if strict_parentheses:
        line = remove_strict_parentheses(line)

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

    # Проверяем, является ли это специальным файлом
    strict_parentheses = source.name == "РАЗВИТИЕ КАПИТАЛИЗМА В РОССИИ.txt"

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:

        # Удаляем технические разделители
        if is_separator(line):
            continue

        line = clean_line(
            line,
            strict_parentheses=strict_parentheses
        )

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