import os
import csv
import re

# Автоматически определяем корень проекта (директорию, где лежит этот скрипт)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "supertest_texts_table.csv")
CLEAN_TEST_DIR = os.path.join(BASE_DIR, "texts", "clean", "test")

# Проверяем, существует ли целевая папка
if not os.path.exists(CLEAN_TEST_DIR):
    print(f"Ошибка: Директория {CLEAN_TEST_DIR} не найдена.")
    print(f"Текущая папка скрипта: {BASE_DIR}")
    exit()

# Шаг 1: Читаем файлы из папки clean/test
files_in_dir = [f for f in os.listdir(CLEAN_TEST_DIR) if f.endswith('.txt')]
# Сортируем для последовательной записи
files_in_dir.sort()

# Получаем имя папки, в которой лежат файлы (в данном случае 'test')
folder_author = os.path.basename(CLEAN_TEST_DIR)

new_rows = []

for fname in files_in_dir:
    # Получаем имя файла без расширения .txt (например, 'testone_01')
    text_id = os.path.splitext(fname)[0]
    
    # Для тестовых файлов номер сегмента и общее количество всегда равны 1
    segment_number = 1
    segment_count = 1
    
    # Автор теперь строго равен названию папки (из 'clean/test' -> 'test')
    author = folder_author
    
    file_path = os.path.join(CLEAN_TEST_DIR, fname)
    
    # Читаем содержимое файла сегмента
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Не удалось прочитать файл {fname}: {e}")
        continue
        
    text_content = "".join(lines)
    
    # Считаем метрики
    line_count = len(lines)
    nonempty_line_count = sum(1 for line in lines if line.strip())
    
    # Подсчет предложений (базовый сплит по знакам препинания)
    sentences = re.split(r'[.!?]+', text_content)
    sentence_count = sum(1 for s in sentences if s.strip())
    
    # Подсчет слов и символов
    words = re.findall(r'\b\w+\b', text_content)
    word_count = len(words)
    char_count = len(text_content)
    
    # Формируем id сегмента строго по вашему образцу: 'testone_01_01'
    segment_id = f"{text_id}_{str(segment_number).zfill(2)}"
    
    # Константные значения по вашему ТЗ
    quadrant = "Q1"
    work = "-"
    genre = "-"
    style = "-"
    split = "test"
    
    # Собираем строку для CSV
    row = [
        segment_id, text_id, segment_number, segment_count, author, fname,
        line_count, nonempty_line_count, sentence_count, word_count, char_count,
        quadrant, work, genre, style, split
    ]
    new_rows.append(row)

# Шаг 2: ДОписываем данные в CSV-файл (режим 'a' - append)
file_exists = os.path.exists(CSV_PATH)

with open(CSV_PATH, "a", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    
    # Если файл создается с нуля, добавим заголовки
    if not file_exists or os.stat(CSV_PATH).st_size == 0:
        headers = [
            "segment_id", "text_id", "segment_number", "segment_count", "author", "filename",
            "line_count", "nonempty_line_count", "sentence_count", "word_count", "char_count",
            "quadrant", "work", "genre", "style", "split"
        ]
        writer.writerow(headers)
        
    writer.writerows(new_rows)

print(f"Успешно обработана папка: {CLEAN_TEST_DIR}")
print(f"Добавлено строк в таблицу '{os.path.basename(CSV_PATH)}': {len(new_rows)}")
