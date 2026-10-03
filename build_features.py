"""
Собирает признаки из двух источников:
  1. clean/*.txt     → 12 существующих метрик (stylometry.extract_features)
  2. processed/*.txt → POS-доли, POS-счётчики, POS N-граммы
"""

from pathlib import Path
import csv

from stylometry import extract_features
from stylometry_processed import (
    extract_processed_features,
    collect_top_pos_ngrams,
    POS_NGRAM_TOP,
)

BASE_DIR      = Path(__file__).resolve().parent / "political_corpus"
CLEAN_DIR     = BASE_DIR / "texts" / "clean"
PROCESSED_DIR = BASE_DIR / "texts" / "processed"
FEATURES_CSV  = BASE_DIR / "features.csv"

TOP_N_GRAMS = 20


def main():
    clean_files = list(CLEAN_DIR.rglob("*.txt"))
    if not clean_files:
        print("Нет файлов в texts/clean")
        return

    processed_paths = [PROCESSED_DIR / f.relative_to(CLEAN_DIR) for f in clean_files]
    processed_paths = [p for p in processed_paths if p.exists()]

    print(f"Собираем топ-{TOP_N_GRAMS} POS N-грамм по корпусу...")
    top = collect_top_pos_ngrams(processed_paths, top_k=TOP_N_GRAMS)
    POS_NGRAM_TOP.update(top)
    print(f"  Униграмм:  {len(top[1])}")
    print(f"  Биграмм:   {len(top[2])}")
    print(f"  Триграмм:  {len(top[3])}")
    print(f"  4-грамм:   {len(top[4])}")

    rows = []
    for clean_path in clean_files:
        rel = clean_path.relative_to(CLEAN_DIR)
        proc_path = PROCESSED_DIR / rel

        feats = extract_features(clean_path)

        if proc_path.exists():
            try:
                feats.update(extract_processed_features(proc_path))
            except Exception as e:
                print(f"  ⚠ processed не прочитан для {rel}: {e}")
        else:
            print(f"  ⚠ нет processed для {rel}")

        feats["file"]   = str(rel)
        feats["author"] = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        rows.append(feats)
        print(f"✓ {rel}")

    fieldnames = list(rows[0].keys())
    with FEATURES_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nГотово: {FEATURES_CSV}")
    print(f"Текстов: {len(rows)}")
    print(f"Признаков на текст: {len(fieldnames)}")


if __name__ == "__main__":
    main()